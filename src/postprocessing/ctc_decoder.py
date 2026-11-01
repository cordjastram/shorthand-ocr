"""
CTC Beam-Search Decoder und Lexikon-/Sprachmodell-Nachkorrektur für Kurzschrift.

Kombiniert:
1. Prefix Beam-Search über die CTC Log-Wahrscheinlichkeiten
2. Historisches deutsches Wortlexikon (inkl. alter Rechtschreibung 1945–1955 & Grundwortschatz)
3. Schnelle Levenshtein-/Subsequenz-Korrektur für typische CTC-Auslassungsfehler
   (z. B. "brckn" -> "brücken", "wiederaufba" -> "wiederaufbau", "dlden" -> "dulden")
"""

import re
import math
from typing import List, Dict, Tuple, Set, Optional
import torch

from src.recognition.vocabulary import DekVocabulary
from src.generator.historical.corpus_loader import HistoricalCorpusLoader, HISTORICAL_DOCUMENTS


class HistoricalLexiconCorrector:
    """Korrigiert erkannte Kurzschrift-Wörter anhand eines historischen deutschen Vokabulars."""

    def __init__(self, extra_words: Optional[List[str]] = None):
        self.words: Set[str] = set()
        self.word_freq: Dict[str, int] = {}
        self._build_default_lexicon()
        if extra_words:
            for w in extra_words:
                self._add_word(w, weight=2)

    def _add_word(self, word: str, weight: int = 1):
        cleaned = word.strip(".,;:!?\"'()[]{}").lower()
        if len(cleaned) >= 2:
            self.words.add(cleaned)
            self.word_freq[cleaned] = self.word_freq.get(cleaned, 0) + weight

    def _build_default_lexicon(self):
        """Extrahiert alle Wörter aus dem historischen Korpus sowie allgemeinen deutschen Grundwortschatz."""
        loader = HistoricalCorpusLoader()
        sources = (
            list(HISTORICAL_DOCUMENTS)
            + loader.SUBJECTS
            + loader.VERBS
            + loader.CLAUSES
            + loader.CONNECTORS
        )
        for text in sources:
            for token in re.findall(r"[\wüöäßÜÖÄß-]+", text):
                self._add_word(token, weight=5)

        # Häufige allgemeine deutsche Funktions- und Inhaltswörter
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
            self._add_word(w, weight=10)

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

        # Satzzeichen am Rand abtrennen
        m = re.match(r"^([^a-zA-ZäöüßÄÖÜ0-9]*)([a-zA-ZäöüßÄÖÜ0-9-]+)([^a-zA-ZäöüßÄÖÜ0-9]*)$", word)
        if not m:
            return word
        prefix, core, suffix = m.groups()
        lower_core = core.lower()

        if len(lower_core) <= 1 or lower_core in self.words or lower_core.isdigit():
            return word

        # Maximale erlaubte Distanz abhängig von der Wortlänge
        allowed_dist = 1 if len(lower_core) <= 4 else min(max_edit_distance, max(2, len(lower_core) // 4))

        best_candidate = None
        best_score = (allowed_dist + 1, 0)  # (dist, -freq)

        first_char = lower_core[0]
        for cand in self.words:
            if abs(len(cand) - len(lower_core)) > allowed_dist:
                continue
            # Erster oder letzter Buchstabe sollte meist übereinstimmen für hohe Präzision
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

        # Großschreibung vom Original übernehmen
        if core[0].isupper():
            best_candidate = best_candidate.capitalize()

        return f"{prefix}{best_candidate}{suffix}"

    def correct_sentence(self, text: str) -> str:
        """Korrigiert alle Wörter eines transkribierten Satzes."""
        if not text:
            return text
        tokens = text.split()
        corrected = [self.correct_word(tok) for tok in tokens]
        return " ".join(corrected)


class CtcBeamSearchDecoder:
    """
    Prefix Beam-Search Decoder für CTC-Log-Wahrscheinlichkeiten
    mit optionaler nachgelagerter Lexikon-Korrektur.
    """

    def __init__(
        self,
        vocab: DekVocabulary,
        beam_width: int = 8,
        use_lexicon: bool = True,
    ):
        self.vocab = vocab
        self.beam_width = beam_width
        self.corrector = HistoricalLexiconCorrector() if use_lexicon else None

    def decode_log_probs(self, log_probs: torch.Tensor, apply_lexicon: bool = True) -> str:
        """
        Führt Prefix Beam-Search auf einem Log-Prob-Tensor der Form (Time, Num_Classes) aus.
        """
        if log_probs.dim() == 3:
            log_probs = log_probs.squeeze(1)

        probs_np = log_probs.detach().cpu()
        time_steps, num_classes = probs_np.shape
        blank = self.vocab.blank_idx

        # Beams: dict[prefix_tuple] -> (p_blank, p_non_blank) in Log-Space
        neg_inf = -1e9
        beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, neg_inf)}

        def log_add(a: float, b: float) -> float:
            if a <= neg_inf:
                return b
            if b <= neg_inf:
                return a
            return max(a, b) + math.log1p(math.exp(-abs(a - b)))

        # Top-K Klassen pro Zeitschritt betrachten (beschleunigt Beam-Search stark)
        topk_k = min(6, num_classes)
        top_vals, top_indices = torch.topk(probs_np, k=topk_k, dim=-1)

        for t in range(time_steps):
            next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}
            t_vals = top_vals[t].tolist()
            t_idxs = top_indices[t].tolist()

            for prefix, (p_b, p_nb) in beams.items():
                p_total = log_add(p_b, p_nb)

                for log_p, c in zip(t_vals, t_idxs):
                    if c == blank:
                        nb_pb, nb_pnb = next_beams.get(prefix, (neg_inf, neg_inf))
                        next_beams[prefix] = (log_add(nb_pb, p_total + log_p), nb_pnb)
                    else:
                        end_c = prefix[-1] if prefix else None
                        new_prefix = prefix + (c,)

                        if c == end_c:
                            # 1) Wiederholung ohne Blank dazwischen -> kollabiert zu prefix
                            cur_pb, cur_pnb = next_beams.get(prefix, (neg_inf, neg_inf))
                            next_beams[prefix] = (cur_pb, log_add(cur_pnb, p_nb + log_p))

                            # 2) Wiederholung nach einem Blank -> erweitert zu new_prefix
                            new_pb, new_pnb = next_beams.get(new_prefix, (neg_inf, neg_inf))
                            next_beams[new_prefix] = (new_pb, log_add(new_pnb, p_b + log_p))
                        else:
                            new_pb, new_pnb = next_beams.get(new_prefix, (neg_inf, neg_inf))
                            next_beams[new_prefix] = (new_pb, log_add(new_pnb, p_total + log_p))

            # Beschneide auf die besten beam_width Präfixe
            sorted_beams = sorted(
                next_beams.items(),
                key=lambda kv: log_add(kv[1][0], kv[1][1]),
                reverse=True,
            )
            beams = dict(sorted_beams[: self.beam_width])

        best_prefix = max(beams.items(), key=lambda kv: log_add(kv[1][0], kv[1][1]))[0]
        raw_text = "".join(
            self.vocab.idx_to_char.get(idx, "")
            for idx in best_prefix
            if idx != blank and self.vocab.idx_to_char.get(idx) != "<unk>"
        )

        if apply_lexicon and self.corrector is not None:
            return self.corrector.correct_sentence(raw_text)
        return raw_text

