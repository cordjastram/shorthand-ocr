# Kurzschrift-Erkennung (DEK - Deutsche Einheitskurzschrift)

Automatisiertes System zur Erkennung handgeschriebener Kurzschrift (DEK Verkehrsschrift) auf gescannten Dokumenten mittels Computer Vision und Deep Learning.

---

## 📁 Projektstruktur

```text
Kurzschrift/
├── data/
│   ├── raw_scans/          # Eingescannte Originaldokumente (TIFF / PNG, 400-600 DPI)
│   ├── processed/          # Entzerrte und segmentierte Zeilenbilder
│   ├── synthetic/          # Synthetisch erzeugte DEK-Trainingsdaten
│   │   ├── demo/           # Visuelle Vergleichsbilder (Clean vs. Handschrift)
│   │   ├── images/         # Datensatz-Bilder
│   │   └── train_manifest.csv # Zuordnung Bilddatei <-> Transkription
│   └── annotations/        # Ground-Truth Transkriptionen für echte Scans
├── src/
│   ├── dek_rules/          # DEK-Grapheme, Stufen, Vokalverbindungen & Kürzel
│   │   ├── symbols.py      # Vektordefinitionen der Grundzeichen & Schattierungen
│   │   └── parser.py       # Phonetischer Text-zu-Stenogramm Parser
│   ├── generator/          # Synthetische Datengenerierung & Augmentierung
│   │   ├── stroke_model.py # Catmull-Rom Spline-Interpolation & Linienzusammenbau
│   │   ├── renderer.py     # Pillow- & SVG-Rendering mit DEK-Hilfslinien
│   │   ├── page_generator.py # Vollständige Stenoblock-Manuskriptseiten
│   │   ├── historical/     # Nachkriegszeit-Spezialisierung (1945–1955)
│   │   │   ├── corpus_loader.py       # Historische Texte & Vokabular (alte Rechtschreibung)
│   │   │   ├── post_war_augmenter.py  # Bleistift-Shader, Holzschliffpapier, Stockflecken, Falten
│   │   │   └── batch_pipeline.py      # Parallele High-Throughput Batch-Generierung
│   │   └── dataset.py      # Zeilen-Batch-Generator
│   ├── preprocessing/      # Scan-Vorverarbeitung
│   │   ├── deskew.py       # Automatische Schräglagenkorrektur (Projektionsvarianz)
│   │   ├── enhancer.py     # CLAHE & Ausleuchtungs-Normalisierung
│   │   ├── segmenter.py    # Zeilensegmentierung über Peak-Tal-Erkennung
│   │   └── pipeline.py     # Master-Pipeline für Dokumente
│   ├── recognition/        # HTR-Erkennungs-Engine
│   │   ├── vocabulary.py   # Zeichensatz & CTC-Dekodierung
│   │   ├── dataset.py      # PyTorch Dataset mit variabler Zeilenbreite
│   │   ├── model.py        # DEK-CRNN (CNN + BiLSTM + CTC)
│   │   ├── train.py        # GPU-Training auf NVIDIA RTX A500 (FP16)
│   │   └── predict.py      # Inferenz-Skript für Zeilenbilder
│   └── postprocessing/     # (In Vorbereitung) Sprachmodell & Kürzel-Rückübertragung
├── tests/
│   ├── test_generator.py       # Generator-Tests
│   ├── test_recognition.py     # Modell- & Vokabular-Tests
│   └── test_preprocessing.py   # Deskew- & Segmentierungs-Tests
├── process_scan.py         # CLI zur automatischen Vorverarbeitung von Scans
├── generate_samples.py     # Einzelzeichen- & Zeilen-Demo
├── generate_test_page.py   # Ganze DIN A5 Manuskriptseite
├── generate_historical_sample.py # Historische Nachkriegsseite & Batch-Lauf
└── requirements.txt
```

---

## 🚀 Schnelleinstieg: DEK-Generator testen

Der synthetische Generator erzeugt aus Fließtext realistische DEK-Stenogramme mit variabler Strichstärke (Haarstrich vs. Schattierung), Höhenstufen (T-Stufe, H-Stufe) und Handschrift-Effekten (Neigung, Zitter, Papiertextur).

### Demo ausführen
```bash
python generate_samples.py
```

### Eigene Sätze synthetisieren
```python
from src.generator.dataset import DekDatasetGenerator

generator = DekDatasetGenerator(output_dir="data/synthetic")

# Einzelnes Bild generieren (Handschrift-Modus mit Hilfslinien)
img = generator.generate_single("der bundestag und die regierung", apply_augmentation=True)
img.save("beispiel.png")

# Ganzen Batch für das Modelltraining erzeugen
saetze = [
    "das leben ist schön",
    "nicht mit vielen worten",
    "der mann ging nach hause"
]
manifest = generator.generate_dataset(saetze, samples_per_text=5, manifest_filename="dataset.csv")
```

---

## 🧪 Tests ausführen

```bash
python -m unittest tests/test_generator.py
```

---

## 📋 Nächste Schritte

1. **Scan-Vorbereitung:** Originale mit 400–600 DPI als Graustufen/Farbe (PNG/TIFF) scannen und in `data/raw_scans/` ablegen.
2. **Vorverarbeitung (`src/preprocessing/`):** Automatische Entzerrung (*Deskewing*) und Zeilensegmentierung für die Scans.
3. **HTR-Modell:** Training eines Zeilenerkenners (z. B. TrOCR) auf den synthetischen Daten, gefolgt von Fine-Tuning auf den echten Scans.
