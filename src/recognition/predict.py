"""
Inferenz- und Vorhersage-Skript für DEK-Zeilen.

Lädt ein trainiertes CRNN-Modell und transkribiert ein Bild
einer Kurzschrift-Zeile in Text.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.vocabulary import DekVocabulary
from src.recognition.model import DekCRNN, build_htr_model
from src.postprocessing.ctc_decoder import CtcBeamSearchDecoder


class DekRecognizer:
    """Führt Vorhersagen auf einzelnen Zeilenbildern aus (inkl. Beam-Search & Lexikon)."""

    def __init__(
        self,
        model_path: str = "models/dek_crnn_best.pt",
        device: str = None,
        use_beam_search: bool = True,
        use_lexicon: bool = True,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.vocab = DekVocabulary()
        self.use_beam_search = use_beam_search
        self.use_lexicon = use_lexicon
        self.beam_decoder = CtcBeamSearchDecoder(self.vocab, beam_width=8, use_lexicon=use_lexicon)
        self.target_height = 64

        model_file = Path(model_path)
        if model_file.exists():
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            arch = checkpoint.get("arch", "crnn")
            self.target_height = checkpoint.get("target_height", 64)
            self.model = build_htr_model(arch=arch, num_classes=len(self.vocab), in_channels=1)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Modellgewichte geladen: {model_file} (Arch: {arch}, Epoche {checkpoint.get('epoch', '?')})")
        else:
            self.model = DekCRNN(num_classes=len(self.vocab), in_channels=1, rnn_hidden_size=256)
            print(f"Hinweis: Keine gespeicherten Gewichte unter {model_file} gefunden. Nutze untrainiertes Modell.")

        self.model.to(self.device)
        self.model.eval()

    def recognize_line(self, image_path: str) -> str:
        """Erkennt den Text in einem Zeilenbild."""
        img = Image.open(image_path).convert("L")

        w, h = img.size
        target_height = self.target_height
        scale = target_height / float(max(1, h))
        new_w = max(16, int(w * scale))
        # Auf Vielfaches von 4 runden
        if new_w % 4 != 0:
            new_w += (4 - (new_w % 4))

        resized = img.resize((new_w, target_height), Image.Resampling.BILINEAR)
        tensor_img = TF.to_tensor(resized)
        # Invertieren (Tinte = Signal)
        tensor_img = 1.0 - tensor_img
        tensor_img = tensor_img.unsqueeze(0).to(self.device)  # (1, 1, H, W)

        with torch.no_grad():
            log_probs = self.model(tensor_img)  # (Time, 1, Num_Classes)
            if self.use_beam_search:
                text = self.beam_decoder.decode_log_probs(log_probs.squeeze(1), apply_lexicon=self.use_lexicon)
            else:
                preds = log_probs.argmax(dim=-1).squeeze(1).tolist()  # (Time,)
                text = self.vocab.ctc_decode(preds)

        return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erkenne DEK-Zeile aus Bilddatei.")
    parser.add_argument("image_path", type=str, help="Pfad zum Zeilenbild")
    parser.add_argument("--model", type=str, default="models/dek_crnn_best.pt")
    args = parser.parse_args()

    recognizer = DekRecognizer(model_path=args.model)
    result = recognizer.recognize_line(args.image_path)
    print(f"\nErkannter Text: \"{result}\"")
