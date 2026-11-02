"""
Inferenz- und Vorhersage-Skript für Kurzschrift-Zeilen (DEK, Gabelsberger, Stolze-Schrey).

Enthält das "Zero-Training Upgrade-Paket" für maximale Erkennungsgenauigkeit ohne Nachtraining:
1. Adaptive Tinten- & Hintergrund-Normalisierung (Perzentil-Clipping, Line-Removal & Kontrastspreizung)
2. Test-Time Augmentation (TTA) mit Multi-Scale-Logit-Ensembling (verschiedene Schreibweiten & Strichstärken)
3. Shallow-Fusion CTC Beam-Search Decoder (Trie + Bigramm-Sprachmodell + Blank-Penalty + Temperatur)
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.vocabulary import DekVocabulary
from src.recognition.model import DekCRNN, build_htr_model
from src.postprocessing.ctc_decoder import CtcBeamSearchDecoder


class DekRecognizer:
    """
    Führt Vorhersagen auf einzelnen Zeilenbildern aus
    (inkl. Test-Time Augmentation, adaptiver Normalisierung, Beam-Search & Sprachmodell).
    """

    def __init__(
        self,
        model_path: str = "models/dek_resnet_attention_best.pt",
        device: str = None,
        use_beam_search: bool = True,
        use_lexicon: bool = True,
        use_tta: bool = True,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.vocab = DekVocabulary()
        self.use_beam_search = use_beam_search
        self.use_lexicon = use_lexicon
        self.use_tta = use_tta
        self.beam_decoder = CtcBeamSearchDecoder(self.vocab, beam_width=10, use_lexicon=use_lexicon)
        self.target_height = 64

        model_file = Path(model_path)
        if not model_file.exists() and model_path == "models/dek_resnet_attention_best.pt":
            fallback = Path("models/dek_crnn_best.pt")
            if fallback.exists():
                model_file = fallback

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

    @staticmethod
    def normalize_ink_tensor(tensor_img: torch.Tensor, clean_background: bool = False) -> torch.Tensor:
        """
        Adaptive Tinten- und Hintergrund-Normalisierung auf dem invertierten Tensor (1, H, W):
        - Schätzt das Hintergrund-Papierniveau über das 50%-Quantil (Median)
        - Spreizt blasse Bleistift-/Tintenstriche über das 99%-Quantil auf [0.0, 1.0]
        - Optional (clean_background=True): Unterdrückt schwache Hilfslinien und Papierkörnung
        """
        t = tensor_img.clone()
        bg_level = float(torch.quantile(t, 0.50))
        ink_peak = float(torch.quantile(t, 0.995))

        if ink_peak > bg_level + 0.05:
            t = torch.clamp((t - bg_level * 0.65) / (ink_peak - bg_level * 0.65), 0.0, 1.0)

        if clean_background:
            t = torch.where(t > 0.18, t, torch.zeros_like(t))

        return t

    def _build_tta_variants(self, img: Image.Image, base_w: int, target_height: int) -> List[torch.Tensor]:
        """
        Erzeugt Test-Time Augmentation (TTA) Varianten derselben Zeile:
        1. Originalskalierung (1.0x)
        2. Originalskalierung mit adaptiver Hintergrund-/Hilfslinien-Bereinigung + Kontrastspreizung
        3. Leicht gestauchte Breite (0.94x) mit adaptiver Normalisierung
        4. Leicht gedehnte Breite (1.06x) mit adaptiver Normalisierung
        """
        variants: List[torch.Tensor] = []

        configs = [
            (1.00, False, False),  # 1. Roh-Tensor wie im Standard-Lauf
            (1.00, True, True),    # 2. Adaptiv normalisiert & hintergrundbereinigt
            (0.94, True, False),   # 3. Gestaucht (kompensiert breit gezogene Handschrift)
            (1.06, True, False),   # 4. Gedehnt (kompensiert eng gedrängte Handschrift)
        ]

        for scale_x, do_norm, clean_bg in configs:
            w_var = max(16, int(round(base_w * scale_x)))
            if w_var % 4 != 0:
                w_var += (4 - (w_var % 4))

            resized = img.resize((w_var, target_height), Image.Resampling.BILINEAR)
            t = 1.0 - TF.to_tensor(resized)  # (1, H, W)
            if do_norm:
                t = self.normalize_ink_tensor(t, clean_background=clean_bg)
            variants.append(t.unsqueeze(0).to(self.device))  # (1, 1, H, W)

        return variants

    def recognize_line(self, image_path: str) -> str:
        """Erkennt den Text in einem Zeilenbild (optional mit Multi-Scale TTA & Shallow-Fusion Beam-Search)."""
        img = Image.open(image_path).convert("L")

        w, h = img.size
        target_height = self.target_height
        scale = target_height / float(max(1, h))
        base_w = max(16, int(w * scale))
        if base_w % 4 != 0:
            base_w += (4 - (base_w % 4))

        with torch.no_grad():
            if self.use_tta:
                variants = self._build_tta_variants(img, base_w, target_height)
                aligned_probs: List[torch.Tensor] = []
                ref_time_steps = None

                for idx, v_tensor in enumerate(variants):
                    lp = self.model(v_tensor)  # (Time_v, 1, Num_Classes)
                    # In Wahrscheinlichkeitsraum umwandeln für sauberes Ensembling
                    probs = torch.exp(lp).permute(1, 2, 0)  # (1, Num_Classes, Time_v)
                    if ref_time_steps is None:
                        ref_time_steps = probs.shape[2]
                    elif probs.shape[2] != ref_time_steps:
                        probs = F.interpolate(
                            probs, size=ref_time_steps, mode="linear", align_corners=False
                        )
                    # Hauptvariante (Index 0 & 1) stärker gewichten als Skalierungsvarianten
                    weight = 0.35 if idx in (0, 1) else 0.15
                    aligned_probs.append(probs * weight)

                avg_probs = torch.clamp(sum(aligned_probs), min=1e-9)
                # Zurück in (Time, Num_Classes) Log-Space
                log_probs_2d = torch.log(avg_probs).squeeze(0).transpose(0, 1)  # (Time, Num_Classes)
            else:
                resized = img.resize((base_w, target_height), Image.Resampling.BILINEAR)
                tensor_img = (1.0 - TF.to_tensor(resized)).unsqueeze(0).to(self.device)
                log_probs_2d = self.model(tensor_img).squeeze(1)

            if self.use_beam_search:
                text = self.beam_decoder.decode_log_probs(log_probs_2d, apply_lexicon=self.use_lexicon)
            else:
                preds = log_probs_2d.argmax(dim=-1).tolist()
                text = self.vocab.ctc_decode(preds)

        return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erkenne Kurzschrift-Zeile aus Bilddatei.")
    parser.add_argument("image_path", type=str, help="Pfad zum Zeilenbild")
    parser.add_argument("--model", type=str, default="models/dek_resnet_attention_best.pt")
    parser.add_argument("--no-tta", action="store_true", help="Deaktiviert Test-Time Augmentation")
    args = parser.parse_args()

    recognizer = DekRecognizer(model_path=args.model, use_tta=(not args.no_tta))
    result = recognizer.recognize_line(args.image_path)
    print(f"\nErkannter Text: \"{result}\"")
