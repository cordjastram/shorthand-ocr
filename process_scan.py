"""
Skript zur Vorverarbeitung gescannter Kurzschrift-Dokumente.

Führt Schräglagenkorrektur (Deskewing), Kontrastoptimierung und
automatische Zeilensegmentierung auf einem Scan durch.
"""

import sys
import argparse
from pathlib import Path

# UTF-8 auf Windows-Konsole erzwingen
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import ScanPreprocessingPipeline
from src.recognition.predict import DekRecognizer


def main():
    parser = argparse.ArgumentParser(description="Verarbeite und transkribiere gescanntes Kurzschrift-Dokument.")
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default="data/synthetic/historical_pretraining/demo/bonn_1948_bleistift.png",
        help="Pfad zum Scan (TIFF/PNG/JPEG)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Ausgabeverzeichnis für Zeilen und Kontrollbilder",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/stolze_schrey_resnet_attention_best.pt",
        help="Pfad zum trainierten HTR-Modell für die Transkription",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Gerät für die Inferenz (z. B. 'cpu' oder 'cuda')",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Scan-Vorverarbeitung & Kurzschrift-Transkription")
    print("=" * 65)
    print(f"Eingabedatei : {args.image_path}")
    print(f"Zielordner   : {args.output_dir}")
    print(f"Modell       : {args.model}\n")

    pipeline = ScanPreprocessingPipeline(output_dir=args.output_dir)
    res = pipeline.process_document(args.image_path)

    print(f"[OK] Schräglage korrigiert: {res['corrected_angle_deg']}°")
    print(f"[OK] Zeilen extrahiert    : {res['num_lines_extracted']}")
    print(f"[OK] Kontrollbild         : {res['debug_image']}")
    print(f"[OK] Manifest-Datei       : {res['manifest_path']}")

    model_file = Path(args.model)
    recognizer = None
    recognizer_raw = None
    if model_file.exists():
        recognizer = DekRecognizer(
            model_path=str(model_file), device=args.device, use_beam_search=True, use_lexicon=True
        )
        recognizer_raw = DekRecognizer(
            model_path=str(model_file), device=args.device, use_beam_search=True, use_lexicon=False
        )

    print("\n--- Extrahierte Zeilen & HTR-Transkription ---")
    for line in res["lines"]:
        idx = line["line_index"] + 1
        img_p = line["image_path"]
        w, h = line["width"], line["height"]
        if recognizer is not None:
            text_lex = recognizer.recognize_line(img_p)
            text_raw = recognizer_raw.recognize_line(img_p)
            print(f"  Zeile {idx:02d} ({w:3d}x{h:2d} px): \"{text_lex}\"  [Roh-CTC: \"{text_raw}\"]")
        else:
            print(f"  Zeile {idx:02d}: {img_p} ({w}x{h} px)")

    print("\n" + "=" * 65)
    print("Verarbeitung erfolgreich abgeschlossen!")
    print("=" * 65)


if __name__ == "__main__":
    main()
