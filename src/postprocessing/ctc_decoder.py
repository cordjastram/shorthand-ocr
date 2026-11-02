"""
CTC Beam-Search Decoder mit Shallow-Fusion Sprachmodell (Trie + Bigramm),
Blank-Penalty, Temperatur-Skalierung und Lexikon-Nachkorrektur für Kurzschrift.

Enthält das "Zero-Training Upgrade-Paket" zur Steigerung der HTR-Genauigkeit ohne Nachtraining:
1. Prefix-Trie & Wort-Bigramm-Sprachmodell direkt WÄHREND des Beam-Search (Shallow Fusion)
2. Temperatur-Skalierung (T) und Blank-Penalty gegen verschluckte Buchstaben/Vokale
3. Historisches deutsches Wortlexikon (inkl. alter Rechtschreibung 1945–1955 & Grundwortschatz)
4. Schnelle Levenshtein-Korrektur für verbleibende Randfälle
"""

import re
import math
from typing import List, Dict, Tuple, Set, Optional
import torch
import torch.nn.functional as F

from src.recognition.vocabulary import DekVocabulary
from src.generator.historical.corpus_loader import HistoricalCorpusLoader, HISTORICAL_DOCUMENTS


class WordTrieNode:
    """Knoten im Präfixbaum (Trie) zur schnellen Prüfung gültiger deutscher Wortanfänge."""
    __slots__ = ("children", "is_word", "log_prob")

    def __init__(self):
        self.children: Dict[str, "WordTrieNode"] = {}
        self.is_word: bool = False
        self.log_prob: float = -10.0


class HistoricalLexiconCorrector:
    """
    Verwaltet das historische deutsche Wortlexikon, einen Präfix-Trie sowie
    Unigramm- und Bigramm-Übergangswahrscheinlichkeiten für Shallow-Fusion Decoding.
    """

    def __init__(self, extra_words: Optional[List[str]] = None):
        self.words: Set[str] = set()
        self.word_freq: Dict[str, int] = {}
        self.bigram_counts: Dict[Tuple[str, str], int] = {}
        self.unigram_total: int = 0
        self.trie_root = WordTrieNode()

        self._build_default_lexicon_and_lm()
        if extra_words:
            for w in extra_words:
                self._add_word(w, weight=3)
        self._rebuild_trie()

    def _add_word(self, word: str, weight: int = 1):
        cleaned = word.strip(".,;:!?\"'()[]{}").lower()
        if len(cleaned) >= 1:
            self.words.add(cleaned)
            self.word_freq[cleaned] = self.word_freq.get(cleaned, 0) + weight
            self.unigram_total += weight

    def _add_sentence_bigrams(self, text: str, weight: int = 3):
        tokens = [
            t.strip(".,;:!?\"'()[]{}").lower()
            for t in re.findall(r"[\wüöäßÜÖÄß-]+", text)
        ]
        tokens = [t for t in tokens if t]
        for tok in tokens:
            self._add_word(tok, weight=weight)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            self.bigram_counts[pair] = self.bigram_counts.get(pair, 0) + weight

    def _build_default_lexicon_and_lm(self):
        """Extrahiert Wörter und Wort-Bigramme aus dem historischen Korpus sowie allgemeinem Grundwortschatz."""
        loader = HistoricalCorpusLoader()
        sources = (
            list(HISTORICAL_DOCUMENTS)
            + loader.SUBJECTS
            + loader.VERBS
            + loader.CLAUSES
            + loader.CONNECTORS
        )
        for text in sources:
            self._add_sentence_bigrams(text, weight=5)

        # Synthetische Beispielsätze samplen, um Satzübergänge (Subjekt -> Verb -> Nebensatz) im Bigramm-LM zu erfassen
        for _ in range(150):
            s = loader.generate_synthetic_sentence()
            self._add_sentence_bigrams(s, weight=2)

        common_words = [
            "und", "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einem", "einen", "eines",
            "nicht", "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden", "hat", "haben", "hatte",
            "mit", "für", "auf", "im", "in", "an", "am", "aus", "bei", "nach", "von", "vom", "zu", "zum", "zur",
            "über", "unter", "durch", "gegen", "ohne", "um", "vor", "zwischen", "während", "wegen", "trotz",
            "daß", "dass", "muß", "muss", "müssen", "kann", "können", "soll", "sollen", "will", "wollen", "darf",
            "schluß", "anschluss", "anschluß", "beschluß", "beschluss", "erläßt", "läßt", "vermißte",
            "freiheit", "frieden", "gerechtigkeit", "menschen", "volk", "regierung", "bundestag", "welt",
            "leben", "arbeit", "wohnung", "städte", "brücken", "wiederaufbau", "militärregierung", "dulden",
            "lebensmittelkarten", "flüchtlingsverwaltung", "alliiertenkontrollrat", "heute", "preise",
            "grundnahrungsmittel", "verfügt", "verfügung", "anordnung", "sitzung", "ordnung", "würde",
        ]
        for w in common_words:
            self._add_word(w, weight=12)

    def _rebuild_trie(self):
        """Baut den Präfix-Trie über alle bekannten Wörter auf."""
        self.trie_root = WordTrieNode()
        total = max(1, self.unigram_total)
        for w, freq in self.word_freq.items():
            node = self.trie_root
            for ch in w:
                if ch not in node.children:
                    node.children[ch] = WordTrieNode()
                node = node.children[ch]
            node.is_word = True
            node.log_prob = math.log(freq / total)

    def is_valid_prefix(self, prefix: str) -> bool:
        """Prüft in O(L), ob ein Wortanfang im Präfix-Trie existiert."""
        if not prefix:
            return True
        node = self.trie_root
        for ch in prefix.lower():
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def get_word_transition_score(self, prev_word: Optional[str], current_word: str) -> float:
        """
        Liefert einen Log-Wahrscheinlichkeits-Score für das abgeschlossene Wort
        unter Berücksichtigung des vorhergehenden Wortes (Bigramm + Unigramm Interpolation).
        """
        w = current_word.strip(".,;:!?\"'()[]{}").lower()
        if not w:
            return 0.0

        total = max(1, self.unigram_total)
        uni_freq = self.word_freq.get(w, 0)
        # Glättung für unbekannte Wörter (OOV Penalty)
        p_uni = (uni_freq + 0.1) / (total + 100.0)

        if prev_word:
            pw = prev_word.strip(".,;:!?\"'()[]{}").lower()
            prev_freq = self.word_freq.get(pw, 0)
            bi_freq = self.bigram_counts.get((pw, w), 0)
            if prev_freq > 0 and bi_freq > 0:
                p_bi = bi_freq / float(prev_freq)
                p_combined = 0.65 * p_bi + 0.35 * p_uni
                return math.log(max(1e-8, p_combined))

        return math.log(max(1e-8, p_uni))

    @staticmethod
    def _edit_distance(s1: str, s2: str, max_dist: int = 3) -> int:
        """Berechnet die Levenshtein-Distanz mit Early-Stopping."""
        if abs(len(s1) - len(s2)) > max_dist:
            return max_dist + 1

        dp = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, start=1):
            new_dp = [i]
            row_min = i
            for j, c2 in enumerate(s2, start=1):
                cost = 0 if c1 == c2 else 1
                val = min(
                    new_dp[-1] + 1,      # Insertion
                    dp[j] + 1,           # Deletion
                    dp[j - 1] + cost,    # Substitution
                )
                new_dp.append(val)
                if val < row_min:
                    row_min = val
            if row_min > max_dist:
                return max_dist + 1
            dp = new_dp
        return dp[-1]

    def correct_word(self, word: str, max_edit_distance: int = 2) -> str:
        """Korrigiert ein einzelnes Wort anhand des Lexikons unter Erhalt von Satzzeichen/Großschreibung."""
        if not word:
            return word

        m = re.match(r"^([^a-zA-ZäöüßÄÖÜ0-9]*)([a-zA-ZäöüßÄÖÜ0-9-]+)([^a-zA-ZäöüßÄÖÜ0-9]*)$", word)
        if not m:
            return word
        prefix, core, suffix = m.groups()
        lower_core = core.lower()

        if len(lower_core) <= 1 or lower_core in self.words or lower_core.isdigit():
            return word

        allowed_dist = 1 if len(lower_core) <= 4 else min(max_edit_distance, max(2, len(lower_core) // 4))

        best_candidate = None
        best_score = (allowed_dist + 1, 0)  # (dist, -freq)

        first_char = lower_core[0]
        for cand in self.words:
            if abs(len(cand) - len(lower_core)) > allowed_dist:
                continue
            if cand[0] != first_char and cand[-1] != lower_core[-1]:
                continue

            dist = self._edit_distance(lower_core, cand, max_dist=allowed_dist)
            if dist <= allowed_dist:
                score = (dist, -self.word_freq.get(cand, 1))
                if score < best_score:
                    best_score = score
                    best_candidate = cand

        if best_candidate is None:
            return word

        if core[0].isupper():
            best_candidate = best_candidate.capitalize()

        return f"{prefix}{best_candidate}{suffix}"

    def correct_sentence(self, text: str, use_bigram_context: bool = True) -> str:
        """Korrigiert alle Wörter eines transkribierten Satzes."""
        if not text:
            return text
        tokens = text.split()
        corrected = [self.correct_word(tok) for tok in tokens]
        return " ".join(corrected)


class CtcBeamSearchDecoder:
    """
    Prefix Beam-Search Decoder mit Shallow-Fusion Sprachmodell (Trie + Bigramm-LM),
    Blank-Penalty und Temperatur-Skalierung.
    """

    def __init__(
        self,
        vocab: DekVocabulary,
        beam_width: int = 10,
        use_lexicon: bool = True,
        lm_weight: float = 0.25,
        word_bonus: float = 0.15,
        prefix_bonus: float = 0.05,
        blank_penalty: float = 0.15,
        temperature: float = 1.15,
    ):
        self.vocab = vocab
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.word_bonus = word_bonus
        self.prefix_bonus = prefix_bonus
        self.blank_penalty = blank_penalty
        self.temperature = temperature
        self.corrector = HistoricalLexiconCorrector() if use_lexicon else None
        self.space_idx = self.vocab.char_to_idx.get(" ", -1)

    def _preprocess_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Wendet Temperatur-Skalierung und Blank-Penalty auf die rohen CTC-Log-Probs an,
        damit schwache Vokale/Verbindungsstriche nicht vom <blank>-Token unterdrückt werden.
        """
        scaled = log_probs.clone()
        if self.blank_penalty > 0.0:
            scaled[:, self.vocab.blank_idx] -= self.blank_penalty
        if self.temperature != 1.0 and self.temperature > 0.0:
            scaled = F.log_softmax(scaled / self.temperature, dim=-1)
        else:
            scaled = F.log_softmax(scaled, dim=-1)
        return scaled

    def _prefix_to_text(self, prefix: Tuple[int, ...]) -> str:
        blank = self.vocab.blank_idx
        return "".join(
            self.vocab.idx_to_char.get(idx, "")
            for idx in prefix
            if idx != blank and self.vocab.idx_to_char.get(idx) != "<unk>"
        )

    def decode_log_probs(self, log_probs: torch.Tensor, apply_lexicon: bool = True) -> str:
        """
        Führt Shallow-Fusion Prefix Beam-Search auf einem Log-Prob-Tensor der Form (Time, Num_Classes) aus.
        """
        if log_probs.dim() == 3:
            log_probs = log_probs.squeeze(1)

        probs_cpu = self._preprocess_log_probs(log_probs.detach().cpu())
        time_steps, num_classes = probs_cpu.shape
        blank = self.vocab.blank_idx

        # Beams: dict[prefix_tuple] -> (p_blank, p_non_blank, lm_score)
        neg_inf = -1e9
        beams: Dict[Tuple[int, ...], Tuple[float, float, float]] = {(): (0.0, neg_inf, 0.0)}

        def log_add(a: float, b: float) -> float:
            if a <= neg_inf:
                return b
            if b <= neg_inf:
                return a
            return max(a, b) + math.log1p(math.exp(-abs(a - b)))

        topk_k = min(8, num_classes)
        top_vals, top_indices = torch.topk(probs_cpu, k=topk_k, dim=-1)

        for t in range(time_steps):
            next_beams: Dict[Tuple[int, ...], Tuple[float, float, float]] = {}
            t_vals = top_vals[t].tolist()
            t_idxs = top_indices[t].tolist()

            for prefix, (p_b, p_nb, lm_s) in beams.items():
                p_total = log_add(p_b, p_nb)

                for log_p, c in zip(t_vals, t_idxs):
                    if c == blank:
                        nb_pb, nb_pnb, nb_lms = next_beams.get(prefix, (neg_inf, neg_inf, lm_s))
                        next_beams[prefix] = (log_add(nb_pb, p_total + log_p), nb_pnb, nb_lms)
                    else:
                        end_c = prefix[-1] if prefix else None
                        new_prefix = prefix + (c,)

                        if c == end_c:
                            # 1) Wiederholung ohne Blank -> kollabiert zu prefix
                            cur_pb, cur_pnb, cur_lms = next_beams.get(prefix, (neg_inf, neg_inf, lm_s))
                            next_beams[prefix] = (cur_pb, log_add(cur_pnb, p_nb + log_p), cur_lms)

                            # 2) Wiederholung nach Blank -> erweitert zu new_prefix
                            new_lms = self._compute_step_lm_score(new_prefix, c, lm_s) if apply_lexicon else 0.0
                            new_pb, new_pnb, _ = next_beams.get(new_prefix, (neg_inf, neg_inf, new_lms))
                            next_beams[new_prefix] = (new_pb, log_add(new_pnb, p_b + log_p), new_lms)
                        else:
                            new_lms = self._compute_step_lm_score(new_prefix, c, lm_s) if apply_lexicon else 0.0
                            new_pb, new_pnb, _ = next_beams.get(new_prefix, (neg_inf, neg_inf, new_lms))
                            next_beams[new_prefix] = (new_pb, log_add(new_pnb, p_total + log_p), new_lms)

            # Sortiere nach kombiniertem CTC + Shallow-Fusion LM-Score
            sorted_beams = sorted(
                next_beams.items(),
                key=lambda kv: log_add(kv[1][0], kv[1][1]) + kv[1][2],
                reverse=True,
            )
            beams = dict(sorted_beams[: self.beam_width])

        best_prefix = max(
            beams.items(),
            key=lambda kv: log_add(kv[1][0], kv[1][1]) + kv[1][2],
        )[0]
        raw_text = self._prefix_to_text(best_prefix)

        if apply_lexicon and self.corrector is not None:
            return self.corrector.correct_sentence(raw_text)
        return raw_text

    def _compute_step_lm_score(self, new_prefix: Tuple[int, ...], added_idx: int, prev_lm_score: float) -> float:
        """
        Bewertet inkrementell den neuen Buchstaben im Präfixbaum (Trie) bzw. bei Leerzeichen
        über das Bigramm-Sprachmodell.
        """
        if self.corrector is None:
            return prev_lm_score

        text = self._prefix_to_text(new_prefix)
        if not text:
            return prev_lm_score

        # Wortgrenze erreicht (Leerzeichen getippt) -> Bigramm-Score für das abgeschlossene Wort addieren
        if added_idx == self.space_idx:
            words = text.strip().split()
            if not words:
                return prev_lm_score
            last_word = words[-1]
            prev_word = words[-2] if len(words) >= 2 else None
            trans_log_p = self.corrector.get_word_transition_score(prev_word, last_word)
            return prev_lm_score + self.lm_weight * trans_log_p + self.word_bonus

        # Mitten im Wort -> Prüfe, ob das aktuelle Wortpräfix im deutschen Trie existiert
        last_space = text.rfind(" ")
        current_word_prefix = text[last_space + 1 :] if last_space != -1 else text
        if self.corrector.is_valid_prefix(current_word_prefix):
            return prev_lm_score + self.prefix_bonus
        else:
            return prev_lm_score - self.prefix_bonus
