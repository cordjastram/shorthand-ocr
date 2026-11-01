"""
Demo- und Testskript für den DEK-Generator.

Generiert Beispielbilder für:
1. Grundzeichen und Kürzel der Verkehrsschrift ("der", "die", "und", "nicht", etc.)
2. Ganze Wörter ("leben", "mut", "baum")
3. Vollständige kurze Sätze (ideal vs. handschriftlich augmentiert)
"""

import sys
from pathlib import Path

# UTF-8 Encoding auf Windows-Konsole sicherstellen
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Sicherstellen, dass das Projektverzeichnis im PYTHONPATH ist
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.generator.dataset import DekDatasetGenerator
from src.dek_rules.parser import DekParser


def main():
    print("=" * 60)
    print("DEK (Deutsche Einheitskurzschrift) - Generator Demo")
    print("=" * 60)

    generator = DekDatasetGenerator(output_dir="data/synthetic")
    parser = DekParser()

    # 1. Testwörter und Sätze
    test_samples = [
        # Kürzel-Tests
        "und",
        "der",
        "die",
        "das",
        "nicht",
        "ist",
        "mit",
        "für",
        # Wort-Tests mit Vokalverbindungen
        "leben",
        "tag",
        "baum",
        "mut",
        # Kurze Phrasen
        "der mann und die welt",
        "nicht für immer",
        "leben mit mut"
    ]

    print("\n[1] Teste Parser auf Beispielsätze:")
    for text in test_samples:
        tokens = parser.parse_text(text)
        print(f"  • '{text}' -> {len(tokens)} Token: {[t.raw_word for t in tokens]}")

    print("\n[2] Generiere visuelle Beleg-Bilder...")
    output_dir = Path("data/synthetic/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, text in enumerate(test_samples):
        # 1. Ideale Version (ohne Verzerrung)
        clean_img = generator.generate_single(text, apply_augmentation=False, draw_guidelines=True)
        clean_file = output_dir / f"sample_{i:02d}_clean.png"
        if clean_img:
            clean_img.save(clean_file)

        # 2. Handschriftlich augmentierte Version (mit Zitter, Neigung, Papiertextur)
        aug_img = generator.generate_single(text, apply_augmentation=True, draw_guidelines=True)
        aug_file = output_dir / f"sample_{i:02d}_handwriting.png"
        if aug_img:
            aug_img.save(aug_file)

        print(f"  [OK] Gespeichert: {clean_file.name} / {aug_file.name} ('{text}')")

    print("\n[3] Generiere einen Mini-Trainingsdatensatz (CSV + Bilder)...")
    dataset_texts = [
        "und", "der", "die", "das", "nicht", "mit", "für", "ist",
        "leben", "baum", "tag", "mut",
        "der bund und die welt",
        "das leben ist gut",
        "nicht mit worten"
    ]
    manifest = generator.generate_dataset(dataset_texts, samples_per_text=2, manifest_filename="train_manifest.csv")

    print("\n" + "=" * 60)
    print(f"Erfolgreich abgeschlossen! {len(manifest)} Trainingsbilder erzeugt in data/synthetic/")
    print("=" * 60)


if __name__ == "__main__":
    main()
