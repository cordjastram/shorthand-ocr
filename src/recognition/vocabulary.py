"""
Vokabular und Tokenizer für die DEK-Zeichenerkennung (CTC-Loss).

Verwaltet Zeichen-Indices, Sonderzeichen (Umlaute, ß, Satzzeichen)
und die CTC-spezifische Dekodierung (Entfernung von <blank> und Wiederholungen).
"""

from typing import List, Dict, Tuple


# Standard-Zeichensatz für deutsche Texte inklusive historischer Orthografie
DEFAULT_CHARACTERS = (
    " abcdefghijklmnopqrstuvwxyz"
    "äöüß"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ÄÖÜ"
    "0123456789"
    ".,;:!?'-/()\""
)


class DekVocabulary:
    """Verwaltet das Vokabular für CTC-Erkennungsmodelle."""

    def __init__(self, characters: str = DEFAULT_CHARACTERS):
        # Index 0 ist für CTC stets das Blank-Symbol
        self.blank_token = "<blank>"
        self.blank_idx = 0

        self.idx_to_char: Dict[int, str] = {self.blank_idx: self.blank_token}
        self.char_to_idx: Dict[str, int] = {self.blank_token: self.blank_idx}

        for idx, char in enumerate(characters, start=1):
            self.idx_to_char[idx] = char
            self.char_to_idx[char] = idx

        self.unk_idx = len(self.idx_to_char)
        self.idx_to_char[self.unk_idx] = "<unk>"
        self.char_to_idx["<unk>"] = self.unk_idx

    def __len__(self) -> int:
        return len(self.idx_to_char)

    def encode(self, text: str) -> List[int]:
        """Wandelt einen Text in eine Liste von Zeichen-Indizes um."""
        return [self.char_to_idx.get(char, self.unk_idx) for char in text]

    def decode(self, indices: List[int]) -> str:
        """Wandelt Indizes ohne CTC-Bereinigung in Text um."""
        return "".join([self.idx_to_char.get(idx, "") for idx in indices if idx != self.blank_idx])

    def ctc_decode(self, indices: List[int]) -> str:
        """
        Standard CTC Greedy-Dekodierung:
        - Zusammenfassung aufeinanderfolgender doppelter Zeichen
        - Entfernung des <blank> Tokens
        """
        decoded_chars = []
        prev_idx = None

        for idx in indices:
            if idx != prev_idx:
                if idx != self.blank_idx and idx in self.idx_to_char:
                    char = self.idx_to_char[idx]
                    if char != "<unk>":
                        decoded_chars.append(char)
                prev_idx = idx

        return "".join(decoded_chars)
