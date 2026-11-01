"""
Skript zur Erzeugung vollständiger DEK-Testseiten (Stenoblock DIN A5).

Erzeugt:
1. Ein hochauflösendes Seitenbild (PNG) mit mehreren Zeilen Fließtext,
   4-Linien-Raster, roter Randlinie und realistischer Scanneigung.
2. Eine JSON-Metadatendatei mit den genauen Bounding-Boxes und Transkriptionen
   jeder einzelnen Zeile (Ground Truth für Zeilensegmentierung und HTR).
"""

import sys
import json
from pathlib import Path

# Sicherstellen, dass UTF-8 auf Windows-Konsole sauber funktioniert
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Projektverzeichnis zum Pfad hinzufügen
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.generator.page_generator import DekPageGenerator


def main():
    print("=" * 60)
    print("Erzeuge vollständige DEK-Testseite (Stenoblock DIN A5)...")
    print("=" * 60)

    # Typischer deutscher Text für die Kurzschrift-Synthese
    sample_text = (
        "der bundestag und die regierung haben heute über das neue gesetz beraten. "
        "das leben in der stadt ist nicht immer leicht aber mit mut und willen kann man viel erreichen. "
        "nicht für jeden tag gibt es eine einfache antwort aber wir müssen lernen mit verstand zu handeln. "
        "die welt wird nicht durch worte allein verändert sondern durch taten die wir gemeinsam vollbringen. "
        "der mann und die frau traten vor das tor und sahen in die ferne. "
        "ist das leben nicht ein wunder das wir jeden tag neu begreifen dürfen. "
        "für die zukunft müssen wir mutig sein und die aufgaben mit freude anpacken. "
        "alle menschen sind frei und gleich an würde und rechten geboren. "
        "sie sind mit vernunft und gewissen begabt und sollen einander im geiste der brüderlichkeit begegnen."
    )

    generator = DekPageGenerator(
        output_dir="data/synthetic/pages",
        page_width=1200,
        page_height=1700,
        margin_left=110,
        margin_right=110,
        margin_top=130,
        margin_bottom=130,
        line_spacing_px=75.0,
        stufe_height_px=22.0,
        base_thickness_px=1.8,
    )

    page_img, meta = generator.generate_page(
        text=sample_text,
        has_margin_line=True,
        apply_page_skew=True,
        page_skew_angle=0.75,
    )

    # 1. Bild speichern
    img_path = Path("data/synthetic/pages/test_page_01.png")
    page_img.save(img_path)
    print(f"\n[OK] Seitenbild gespeichert: {img_path} ({page_img.width}x{page_img.height} px)")

    # 2. Annotationen (Ground Truth) speichern
    meta_path = Path("data/synthetic/pages/test_page_01.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] Metadaten gespeichert: {meta_path} ({meta['num_lines']} Zeilen)")

    print("\n--- Zeilenübersicht der erzeugten Seite ---")
    for line in meta["lines"]:
        print(f"  Zeile {line['line_index'] + 1:02d} (Baseline y={line['baseline_y']}): \"{line['text']}\"")

    print("\n" + "=" * 60)
    print("Testseite erfolgreich erstellt!")
    print("=" * 60)


if __name__ == "__main__":
    main()
