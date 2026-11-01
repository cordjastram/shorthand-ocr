"""
Kurzschrift Text-zu-Stenogramm Parser (DEK, Gabelsberger, Stolze-Schrey).

Analysiert deutschen Text und zerlegt ihn in Kurzschrift-Elemente:
1. Prüfung auf feste Kürzel/Sigel des gewählten Systems (z. B. "der", "die", "und", "nicht")
2. Zerlegung unbekannter Wörter in Konsonanten und Vokal-Verbindungsglieder
"""

import re
from typing import List, Tuple, Optional, Dict
from src.dek_rules.symbols import CONSONANTS, VOWEL_CONNECTORS, KUERZEL, DekGlyph, StrokeSegment
from src.dek_rules.gabelsberger_symbols import (
    GABELSBERGER_CONSONANTS,
    GABELSBERGER_VOWELS,
    GABELSBERGER_KUERZEL,
)
from src.dek_rules.stolze_schrey_symbols import (
    STOLZE_SCHREY_CONSONANTS,
    STOLZE_SCHREY_VOWELS,
    STOLZE_SCHREY_KUERZEL,
)

SUPPORTED_SYSTEMS = ("dek", "gabelsberger", "stolze_schrey")


def get_system_dictionaries(system: str) -> Tuple[Dict[str, DekGlyph], Dict[str, DekGlyph], Dict[str, DekGlyph]]:
    """Gibt (CONSONANTS, VOWELS, KUERZEL) für das gewählte Kurzschriftsystem zurück."""
    sys_key = system.strip().lower().replace("-", "_")
    if sys_key == "dek":
        return CONSONANTS, VOWEL_CONNECTORS, KUERZEL
    elif sys_key == "gabelsberger":
        return GABELSBERGER_CONSONANTS, GABELSBERGER_VOWELS, GABELSBERGER_KUERZEL
    elif sys_key in ("stolze_schrey", "stolzeschrey"):
        return STOLZE_SCHREY_CONSONANTS, STOLZE_SCHREY_VOWELS, STOLZE_SCHREY_KUERZEL
    else:
        raise ValueError(
            f"Unbekanntes Kurzschriftsystem '{system}'. Unterstützt: {SUPPORTED_SYSTEMS}"
        )


class DekWordToken:
    """Repräsentiert ein einzelnes Wort als Sequenz von Kurzschrift-Graphemen."""
    def __init__(self, raw_word: str, is_kuerzel: bool = False, system: str = "dek"):
        self.raw_word = raw_word
        self.is_kuerzel = is_kuerzel
        self.system = system
        self.glyphs: List[DekGlyph] = []

    def __repr__(self):
        glyph_names = [g.name for g in self.glyphs]
        return f"DekWordToken(word='{self.raw_word}', system='{self.system}', kuerzel={self.is_kuerzel}, glyphs={glyph_names})"


class DekParser:
    """Zerlegt deutschen Fließtext in Kurzschrift-Repräsentationen (DEK, Gabelsberger, Stolze-Schrey)."""

    # Normalisierungsregeln für Lautersetzungen
    REPLACEMENTS = [
        ("sch", "sch"),
        ("ch", "ch"),
        ("ck", "k"),
        ("ph", "f"),
        ("qu", "kw"),
        ("ß", "s"),
        ("ss", "s"),
        ("tz", "z"),
        ("v", "w"),
        ("c", "k"),
    ]

    # Bekannte Vokalcluster (Diphthonge zuerst)
    VOWELS = ["ei", "au", "eu", "äu", "ie", "ö", "ü", "ä", "a", "e", "i", "o", "u"]

    def __init__(self, system: str = "dek"):
        self.system = system.strip().lower().replace("-", "_")
        self.consonants, self.vowels, self.kuerzel = get_system_dictionaries(self.system)

    def parse_text(self, text: str) -> List[DekWordToken]:
        """Zerlegt einen Text in eine Liste von DEK-Wort-Token."""
        words = re.findall(r"\b[\wüöäßÜÖÄß-]+\b", text)
        parsed_tokens = []
        for word in words:
            token = self.parse_word(word)
            if token and token.glyphs:
                parsed_tokens.append(token)
        return parsed_tokens

    def parse_word(self, word: str) -> DekWordToken:
        """Parst ein einzelnes deutsches Wort."""
        cleaned = word.strip().lower()
        if not cleaned:
            return DekWordToken(word, system=self.system)

        # 1. Kürzel-Check
        if cleaned in self.kuerzel:
            token = DekWordToken(raw_word=word, is_kuerzel=True, system=self.system)
            token.glyphs.append(self.kuerzel[cleaned])
            return token

        # 2. Phonem-Zerlegung (Konsonant - Vokal - Konsonant)
        token = DekWordToken(raw_word=word, is_kuerzel=False, system=self.system)
        phonetic_seq = self._tokenize_word_to_phonemes(cleaned)

        # 3. Mappe Phoneme auf DEK-Grapheme
        # In DEK steht ein Konsonant, dann folgt eine Vokalverbindung zum nächsten Konsonanten.
        i = 0
        while i < len(phonetic_seq):
            phoneme = phonetic_seq[i]

            # Ist es ein Vokal am Wortanfang?
            if self._is_vowel(phoneme) and i == 0:
                # Anlaut-Vokal: In DEK oft durch Grundstrich oder direkte Vokalandeutung
                v_glyph = self._get_vowel_glyph(phoneme)
                if v_glyph:
                    token.glyphs.append(v_glyph)
                i += 1
                continue

            # Ist es ein Konsonant?
            if phoneme in self.consonants:
                token.glyphs.append(self.consonants[phoneme])

                # Folgt ein Vokal?
                if i + 1 < len(phonetic_seq) and self._is_vowel(phonetic_seq[i + 1]):
                    vowel_str = phonetic_seq[i + 1]
                    v_glyph = self._get_vowel_glyph(vowel_str)
                    if v_glyph:
                        token.glyphs.append(v_glyph)
                    i += 2
                    continue
                else:
                    # Direkte Konsonantenverbindung ohne Vokal (z. B. "st", "nd", "tr") -> enges 'e'
                    if i + 1 < len(phonetic_seq) and phonetic_seq[i + 1] in self.consonants:
                        token.glyphs.append(self.vowels["e"])
                    i += 1
                    continue

            # Fallback für unbekanntes Zeichen
            i += 1

        return token

    def _is_vowel(self, s: str) -> bool:
        return s in self.VOWELS or s in ["e", "i", "o", "u", "a", "ei", "au"]

    def _get_vowel_glyph(self, vowel_str: str) -> Optional[DekGlyph]:
        """Findet das passende Verbindungs-Graphem für einen Vokal."""
        # Vereinfachung phonetischer Varianten:
        v_map = {
            "ie": "i",
            "ä": "e",
            "ö": "i",
            "ü": "u",
            "eu": "ei",
            "äu": "ei",
        }
        mapped = v_map.get(vowel_str, vowel_str)
        return self.vowels.get(mapped, self.vowels.get("e"))

    def _tokenize_word_to_phonemes(self, word: str) -> List[str]:
        """Zerlegt ein Wort in Segmente aus Konsonanten und Vokalen."""
        # Normalisiere Konsonantenverbindungen
        w = word
        for old, new in self.REPLACEMENTS:
            w = w.replace(old, new)

        tokens = []
        idx = 0
        while idx < len(w):
            # Prüfe Mehrbuchstaben-Elemente zuerst (sch, ch, ei, au, etc.)
            matched = False
            for length in [3, 2, 1]:
                chunk = w[idx:idx + length]
                if chunk in self.consonants or chunk in self.VOWELS:
                    tokens.append(chunk)
                    idx += length
                    matched = True
                    break
            if not matched:
                # Buchstabe überspringen, falls nicht im Alphabet
                idx += 1

        return tokens
