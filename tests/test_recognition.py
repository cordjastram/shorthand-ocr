"""
Automatisierte Tests für das DEK-Erkennungsmodul (CRNN & CTC).
"""

import unittest
from pathlib import Path
import torch
import torch.nn as nn

from src.recognition.vocabulary import DekVocabulary
from src.recognition.model import DekCRNN, DekResNetAttentionHTR, build_htr_model
from src.recognition.dataset import DekLineDataset, ctc_collate_fn
from src.postprocessing.ctc_decoder import CtcBeamSearchDecoder, HistoricalLexiconCorrector


class TestDekRecognition(unittest.TestCase):

    def setUp(self):
        self.vocab = DekVocabulary()

    def test_vocabulary_encoding_decoding(self):
        """Prüft, ob Texte korrekt in Indizes und zurück kodiert werden."""
        text = "der bundestag 1948"
        encoded = self.vocab.encode(text)
        self.assertEqual(len(encoded), len(text))

        # CTC Decode Test mit Wiederholungen und Blanks
        # Simulation: [d, d, blank, e, blank, r, r] -> "der"
        d_idx = self.vocab.char_to_idx["d"]
        e_idx = self.vocab.char_to_idx["e"]
        r_idx = self.vocab.char_to_idx["r"]
        b_idx = self.vocab.blank_idx

        raw_ctc_seq = [d_idx, d_idx, b_idx, e_idx, b_idx, r_idx, r_idx]
        decoded = self.vocab.ctc_decode(raw_ctc_seq)
        self.assertEqual(decoded, "der")

    def test_crnn_forward_pass_and_shapes(self):
        """Prüft den Forward-Pass des Modells und die Dimensionen."""
        num_classes = len(self.vocab)
        model = DekCRNN(num_classes=num_classes, in_channels=1, rnn_hidden_size=128)
        model.eval()

        # Dummy Input: Batch=2, Channel=1, Height=64, Width=200
        x = torch.randn(2, 1, 64, 200)
        with torch.no_grad():
            log_probs = model(x)

        # Erwartete Ausgabeform für CTC: (Time_Steps, Batch_Size, Num_Classes)
        # Width 200 wird durch 2x MaxPool im CNN um Faktor 4 reduziert -> Time_Steps = 50
        time_steps, batch_size, out_classes = log_probs.shape
        self.assertEqual(batch_size, 2)
        self.assertEqual(time_steps, 50)
        self.assertEqual(out_classes, num_classes)

    def test_ctc_loss_calculation(self):
        """Prüft, ob der CTC Loss fehlerfrei berechnet werden kann."""
        num_classes = len(self.vocab)
        model = DekCRNN(num_classes=num_classes, in_channels=1, rnn_hidden_size=128)
        criterion = nn.CTCLoss(blank=self.vocab.blank_idx, zero_infinity=True)

        x = torch.randn(2, 1, 64, 200)
        log_probs = model(x)  # (50, 2, Num_Classes)

        targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
        target_lengths = torch.tensor([3, 3], dtype=torch.long)
        input_lengths = torch.tensor([50, 50], dtype=torch.long)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_resnet_attention_htr_forward_and_beam_decoder(self):
        """Prüft das große ResNet-BiLSTM-Attention Modell (H=96) und den Beam-Search/Lexikon-Decoder."""
        num_classes = len(self.vocab)
        model = build_htr_model("resnet_attention", num_classes=num_classes, in_channels=1)
        model.eval()

        # Input mit erhöhter vertikaler Auflösung H=96, W=240
        x = torch.randn(2, 1, 96, 240)
        with torch.no_grad():
            log_probs = model(x)

        time_steps, batch_size, out_classes = log_probs.shape
        self.assertEqual(batch_size, 2)
        self.assertEqual(time_steps, 60)  # 240 // 4
        self.assertEqual(out_classes, num_classes)

        # Lexikon-Korrektur testen
        corrector = HistoricalLexiconCorrector()
        self.assertEqual(
            corrector.correct_sentence("die Militäregierung an nicht dlden daß der wiederaufba der zerstörten brckn"),
            "die Militärregierung an nicht dulden daß der wiederaufbau der zerstörten brücken",
        )
        # Trie-Präfix und Bigramm-Score prüfen
        self.assertTrue(corrector.is_valid_prefix("wiederauf"))
        self.assertFalse(corrector.is_valid_prefix("xyzqwert"))
        score_known = corrector.get_word_transition_score("der", "wiederaufbau")
        score_unknown = corrector.get_word_transition_score("der", "xyzwort")
        self.assertGreater(score_known, score_unknown)

        # Shallow-Fusion Beam-Search mit Blank-Penalty und Temperatur testen
        decoder = CtcBeamSearchDecoder(self.vocab, beam_width=6, use_lexicon=True)
        decoded_str = decoder.decode_log_probs(log_probs[:, 0, :], apply_lexicon=True)
        self.assertIsInstance(decoded_str, str)


if __name__ == "__main__":
    unittest.main()
