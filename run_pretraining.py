"""
Master-Skript für das historische Kurzschrift-Pretraining auf der NVIDIA RTX A500.

Unterstützt die drei großen deutschen Kurzschriftsysteme:
- DEK (Deutsche Einheitskurzschrift, ab 1924)
- Gabelsberger (Franz Xaver Gabelsberger, ab 1834)
- Stolze-Schrey (Vereinigtes System Stolze-Schrey, ab 1897)

Workflow je System:
1. Parallele Generierung von historischen Manuskriptseiten auf allen CPU-Kernen
   mit Nachkriegsvokabular (1945–1955), Holzschliffpapier, Stockflecken und Bleistifttextur.
2. GPU-Training der CRNN über 25 Epochen mit FP16 Mixed Precision auf der RTX A500.
3. Stichproben-Evaluation auf ungesehenen Testzeilen.
"""

import sys
import time
import argparse
from pathlib import Path

# UTF-8 auf Windows-Konsole sicherstellen
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.generator.historical.batch_pipeline import HistoricalBatchPipeline
from src.recognition.train import train
from src.recognition.predict import DekRecognizer

SYSTEM_CONFIGS = {
    "dek": {
        "label": "DEK (Deutsche Einheitskurzschrift)",
        "output_dir": "data/synthetic/historical_pretraining",
        "model_filename": "dek_crnn_best.pt",
    },
    "gabelsberger": {
        "label": "Gabelsberger-Kurzschrift (1834)",
        "output_dir": "data/synthetic/historical_gabelsberger",
        "model_filename": "gabelsberger_crnn_best.pt",
    },
    "stolze_schrey": {
        "label": "Stolze-Schrey-Kurzschrift (1897)",
        "output_dir": "data/synthetic/historical_stolze_schrey",
        "model_filename": "stolze_schrey_crnn_best.pt",
    },
}


def run_for_system(
    system: str,
    num_pages: int = 500,
    epochs: int = 25,
    arch: str = "resnet_attention",
    target_height: int = 96,
):
    cfg = SYSTEM_CONFIGS[system]
    start_time = time.time()
    print("\n" + "=" * 70)
    print(f"{cfg['label']}: Großes Historisches Pretraining ({arch.upper()}, H={target_height}px)")
    print("=" * 70)

    output_dir = cfg["output_dir"]
    model_filename = cfg["model_filename"]
    if arch != "crnn":
        model_filename = model_filename.replace("_crnn_best.pt", f"_{arch}_best.pt")
    manifest_file = Path(output_dir) / "lines_manifest.csv"

    # 1. Datensatz generieren, falls noch nicht vorhanden
    if manifest_file.exists():
        print(f"\n>>> SCHRITT 1/2: Datensatz bereits vorhanden ({manifest_file}) – überspringe Neugenerierung.")
    else:
        print(f"\n>>> SCHRITT 1/2: Generiere {num_pages} historische Seiten ({system.upper()}) auf allen CPU-Kernen...")
        pipeline = HistoricalBatchPipeline(output_dir=output_dir, system=system)
        summary = pipeline.generate_batch(
            num_pages=num_pages,
            train_ratio=0.8,
            val_ratio=0.1,
        )

        gen_time = time.time() - start_time
        print(f"\n[OK] Phase 1 abgeschlossen in {gen_time / 60:.1f} Minuten.")
        print(f"     -> {summary['num_lines']} Zeilen bereitgestellt in {manifest_file}")

    # 2. GPU-Training starten
    print(f"\n>>> SCHRITT 2/2: Starte GPU-Training ({system.upper()}, {arch}) auf der NVIDIA RTX A500...")
    train_start = time.time()

    train(
        manifest_csv=str(manifest_file),
        models_dir="models",
        epochs=epochs,
        batch_size=12 if target_height >= 96 else 16,
        lr=3e-4,
        use_fp16=True,
        resume=True,
        model_filename=model_filename,
        arch=arch,
        target_height=target_height,
    )

    train_time = time.time() - train_start
    total_time = time.time() - start_time

    # 3. Test-Inferenz mit dem trainierten Modell
    model_path = f"models/{model_filename}"
    print("\n" + "=" * 70)
    print(f"Pretraining ({cfg['label']}) erfolgreich abgeschlossen!")
    print(f"  • Trainingszeit GPU : {train_time / 60:.1f} Minuten")
    print(f"  • Gesamtlaufzeit    : {total_time / 60:.1f} Minuten")
    print(f"  • Gespeichertes Modell : {model_path}")
    print("=" * 70)

    # Kurze Demonstration auf Testzeilen
    print("\n>>> Führe Test-Transkriptionen (mit Beam-Search & Lexikon) auf ungesehenen Zeilen durch:")
    test_lines = list((Path(output_dir) / "lines").glob("*.png"))[:3]
    if test_lines:
        recognizer = DekRecognizer(model_path=model_path, use_beam_search=True, use_lexicon=True)
        for line_p in test_lines:
            text = recognizer.recognize_line(str(line_p))
            print(f"  • Zeile {line_p.name}: \"{text}\"")


def main():
    parser = argparse.ArgumentParser(description="Historisches Kurzschrift-Pretraining (DEK, Gabelsberger, Stolze-Schrey).")
    parser.add_argument(
        "--system",
        type=str,
        default="all",
        choices=["dek", "gabelsberger", "stolze_schrey", "all"],
        help="Welches Kurzschriftsystem trainiert werden soll (Standard: all)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet_attention",
        choices=["crnn", "resnet_attention"],
        help="Modellarchitektur (Standard: resnet_attention)",
    )
    parser.add_argument("--height", type=int, default=96, help="Ziel-Zeilenhöhe in Pixeln (Standard: 96)")
    parser.add_argument("--pages", type=int, default=500, help="Anzahl zu generierender Seiten pro System")
    parser.add_argument("--epochs", type=int, default=25, help="Anzahl der Trainings-Epochen")
    args = parser.parse_args()

    systems_to_run = ["dek", "gabelsberger", "stolze_schrey"] if args.system == "all" else [args.system]
    for sys_name in systems_to_run:
        run_for_system(
            sys_name,
            num_pages=args.pages,
            epochs=args.epochs,
            arch=args.arch,
            target_height=args.height,
        )


if __name__ == "__main__":
    main()
