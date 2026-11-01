# Historische Kurzschrift-Erkennung (DEK, Gabelsberger & Stolze-Schrey)

Automatisiertes System zur Synthese, Vorverarbeitung und Handschrifterkennung (HTR) historischer deutscher Kurzschrift-Manuskripte mittels Computer Vision und Deep Learning (PyTorch, CUDA FP16).

Unterstützt die drei großen deutschen Stenografiesysteme:
1. **DEK (Deutsche Einheitskurzschrift, ab 1924 / Verkehrsschrift)**
2. **Gabelsberger-Kurzschrift (Franz Xaver Gabelsberger, ab 1834)**
3. **Stolze-Schrey-Kurzschrift (Vereinigtes System Stolze-Schrey, ab 1897)**

---

## 📁 Projektstruktur

```text
Kurzschrift/
├── data/
│   ├── raw_scans/          # Eingescannte Originaldokumente (TIFF / PNG / JPG)
│   ├── processed/          # Entzerrte und segmentierte Zeilenbilder + Debug-Overlays
│   ├── synthetic/          # Synthetisch erzeugte Trainingsdaten (DEK, Gabelsberger, Stolze-Schrey)
│   │   ├── historical_pretraining/   # DEK Nachkriegs-Datensatz & Demo-Seiten
│   │   ├── historical_gabelsberger/  # Gabelsberger Nachkriegs-Datensatz
│   │   └── historical_stolze_schrey/ # Stolze-Schrey Nachkriegs-Datensatz
│   └── annotations/        # Ground-Truth Transkriptionen für echte Scans
├── models/                 # Trainierte PyTorch HTR-Modellgewichte (*.pt)
├── src/
│   ├── dek_rules/          # Regelwerke, Grapheme, Vokalverbindungen & Kürzel/Sigel
│   │   ├── symbols.py                # DEK Grundzeichen, Vokalstufen & Kürzel
│   │   ├── gabelsberger_symbols.py   # Gabelsberger (1834) Grapheme, Vokalisation & Sigel
│   │   ├── stolze_schrey_symbols.py  # Stolze-Schrey (1897) Abstrich-Systematik & Kürzel
│   │   └── parser.py                 # Multi-System Text-zu-Stenogramm Parser
│   ├── generator/          # Synthetische Datengenerierung & Augmentierung
│   │   ├── stroke_model.py           # Catmull-Rom Spline-Interpolation & Linienzusammenbau
│   │   ├── renderer.py               # Pillow- & SVG-Rendering mit variabler Druckstärke
│   │   ├── page_generator.py         # Vollständige Manuskriptseiten (liniert & unliniert/reinweiß)
│   │   ├── dataset.py                # Zeilen-Batch-Generator
│   │   └── historical/               # Nachkriegszeit-Spezialisierung (1945–1955)
│   │       ├── corpus_loader.py      # Historische Texte & Vokabular (alte Rechtschreibung)
│   │       ├── post_war_augmenter.py # Bleistift/Eisengallustinte, Holzschliff, Stockflecken, Falten
│   │       └── batch_pipeline.py     # Parallele High-Throughput Multi-Core Batch-Generierung
│   ├── preprocessing/      # Scan-Vorverarbeitung
│   │   ├── deskew.py                 # Automatische Schräglagenkorrektur (Projektionsvarianz)
│   │   ├── enhancer.py               # CLAHE & Ausleuchtungs-Normalisierung
│   │   ├── segmenter.py              # Auflösungsadaptiver Zeilensegmentierer (Peak-Tal-Analyse)
│   │   └── pipeline.py               # Master-Pipeline für Dokumentenscans
│   ├── recognition/        # HTR-Erkennungs-Engine
│   │   ├── vocabulary.py             # Zeichensatz & CTC-Dekodierung
│   │   ├── dataset.py                # PyTorch Dataset (On-the-fly Augmentierung, SpecAugment & Line-Removal)
│   │   ├── model.py                  # DekResNetAttentionHTR (SE-ResNet + BiLSTM + Transformer) & DekCRNN
│   │   ├── train.py                  # GPU-Training (FP16 Mixed Precision, CosineAnnealingLR, Resume)
│   │   └── predict.py                # Inferenz-Klasse (DekRecognizer) mit Beam-Search & Lexikon
│   └── postprocessing/     # Nachbearbeitung & Sprachmodell
│       └── ctc_decoder.py            # CTC Prefix Beam-Search Decoder & Historisches Wortlexikon
├── tests/
│   ├── test_generator.py             # Generator-Tests für DEK, Gabelsberger & Stolze-Schrey
│   ├── test_recognition.py           # Modell-, Beam-Search- & Lexikon-Tests
│   └── test_preprocessing.py         # Deskew- & Segmentierungs-Tests
├── run_pretraining.py                # Master-Skript für Datengenerierung & GPU-Training aller Systeme
├── process_scan.py                   # CLI zur Vorverarbeitung & Transkription echter Scans
├── generate_historical_sample.py     # Erzeugt historische Beispielseiten für alle 3 Systeme
├── generate_samples.py               # Einzelzeichen- & Zeilen-Demo
├── generate_test_page.py             # Ganze DIN A5 Manuskriptseite
└── requirements.txt
```

---

## 🧠 HTR-Modellarchitektur (`DekResNetAttentionHTR`)

Für hohe Erkennungsgenauigkeit bei feinen Kurzschrift-Merkmalen (Haarstrich vs. Druckverstärkung/Schattierung, Halbstufen vs. Ganzstufen) kommt ein hybrides **Vision-Sequence-Attention-Modell** (`src/recognition/model.py`) zum Einsatz:

- **Auflösung:** `Height = 96 px` bei variabler Zeilenbreite.
- **Backbone:** 4-stufiges **ResNet mit Squeeze-and-Excitation (`SqueezeExcitation2d`)** zur präzisen Kanalgewichtung von Strichstärken und Schattierungen.
- **Sequenz-Encoder:** **3-Layer BiLSTM** (`hidden_size=384`, bidirektional -> `768` Kanäle) mit Residual-LayerNorm-Projektion.
- **Globaler Kontext:** **2-Layer Multi-Head Self-Attention Transformer-Encoder** (`8 Heads`, `GELU`, `Sinusoidal Positional Encoding`).
- **Training & Regularisierung:** `CTCLoss`, `AdamW`, `CosineAnnealingLR`, `FP16 Mixed Precision`, On-the-fly **SpecAugment** (Zeit-/Höhenmasken) sowie dynamische Hilfslinien-Unterdrückung (Training für linierte Stenoblöcke und unliniertes weißes Briefpapier).
- **Decoding (`src/postprocessing/ctc_decoder.py`):** **CTC Prefix Beam-Search** kombiniert mit einem historischen deutschen Lexikon-Korrektor (`HistoricalLexiconCorrector`).

---

## 🚀 Schnelleinstieg

### 1. Historische Beispielseiten für alle 3 Kurzschriftsysteme erzeugen
```bash
python generate_historical_sample.py
```
Erzeugt im Ordner `data/synthetic/historical_pretraining/demo/`:
- `bonn_1948_bleistift.png` (DEK)
- `bonn_1948_gabelsberger.png` (Gabelsberger)
- `bonn_1948_stolze_schrey.png` (Stolze-Schrey)

### 2. Großes Pretraining starten (Datengenerierung + GPU-Training)
```bash
# Alle drei Systeme (DEK, Gabelsberger, Stolze-Schrey) mit ResNet-Attention (H=96) trainieren:
python run_pretraining.py --system all --arch resnet_attention --height 96 --pages 500 --epochs 25

# Oder gezielt ein einzelnes System trainieren / fortsetzen (automatisches Checkpoint-Resume):
python run_pretraining.py --system stolze_schrey --arch resnet_attention --height 96 --epochs 25
```

### 3. Echte Scans vorverarbeiten, segmentieren und transkribieren
```bash
python process_scan.py pfad/zum/scan.jpg --output-dir data/processed/mein_scan --model models/stolze_schrey_resnet_attention_best.pt
```
Das Skript führt automatisch folgende Schritte aus:
1. **Deskewing:** Schräglagenkorrektur der gescannten Seite.
2. **CLAHE-Enhancement:** Kontrastnormalisierung.
3. **Adaptive Zeilensegmentierung:** Extrahiert alle Textzeilen unabhängig von der Scan-Auflösung und speichert ein Debug-Kontrollbild (`*_segmented_debug.png`).
4. **HTR-Inferenz:** Transkribiert jede Zeile mittels `DekRecognizer` (inkl. Beam-Search & Lexikon-Nachkorrektur).

### 4. Einzelne Zeilen per Python-API transkribieren
```python
from src.recognition.predict import DekRecognizer

recognizer = DekRecognizer(
    model_path="models/stolze_schrey_resnet_attention_best.pt",
    use_beam_search=True,
    use_lexicon=True,
)
text = recognizer.recognize_line("data/processed/lines/zeile_001.png")
print("Transkription:", text)
```

---

## 🧪 Automatisierte Tests

Alle Module (Regelwerke für DEK/Gabelsberger/Stolze-Schrey, Seiten-Generator, Scan-Segmentierung, ResNet-Attention-HTR und Beam-Search-Decoder) lassen sich über die Test-Suite prüfen:

```bash
python -m unittest discover -s tests -v
```
