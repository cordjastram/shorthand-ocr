"""
Test- und Demonstrationsskript für historische Nachkriegs-DEK-Dokumente (1945–1955).

Erzeugt:
1. Eine Einzelseite mit Text des Parlamentarischen Rates (Bonn 1948)
   mit Holzschliff-Papier, Stockflecken, Randvergilbung und Bleistift-Strichen.
2. Einen Mini-Batch-Lauf über die multiprocessing Batch-Pipeline.
"""

import sys
from pathlib import Path

# UTF-8 auf Windows-Konsole erzwingen
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.generator.page_generator import DekPageGenerator
from src.generator.historical.corpus_loader import HistoricalCorpusLoader
from src.generator.historical.post_war_augmenter import PostWarAugmenter
from src.generator.historical.batch_pipeline import HistoricalBatchPipeline


def generate_single_demo():
    print("=" * 60)
    print("Erzeuge authentische Nachkriegs-Kurzschrift-Seiten (Bonn 1948)...")
    print("=" * 60)

    corpus = HistoricalCorpusLoader()
    text = corpus.get_curated_document(0)  # Parlamentarischer Rat 1948/49

    output_dir = Path("data/synthetic/historical_pretraining/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    systems = [
        ("dek", "bonn_1948_bleistift.png", "aged_yellow", "pencil"),
        ("gabelsberger", "bonn_1948_gabelsberger.png", "faded_cream", "faded_ink"),
        ("stolze_schrey", "bonn_1948_stolze_schrey.png", "wood_pulp_brown", "pencil"),
    ]

    generated_files = []
    for sys_name, filename, paper_tone, tool_type in systems:
        augmenter = PostWarAugmenter(paper_tone=paper_tone, tool_type=tool_type)
        tool_color, thickness = augmenter.get_tool_color_and_thickness()

        generator = DekPageGenerator(
            output_dir=str(output_dir),
            page_width=1200,
            page_height=1700,
            margin_left=110,
            margin_right=110,
            margin_top=130,
            margin_bottom=130,
            line_spacing_px=75.0,
            stufe_height_px=22.0,
            base_thickness_px=thickness,
            system=sys_name,
        )
        generator.renderer.ink_color = tool_color

        page_img, meta = generator.generate_page(
            text=text,
            has_margin_line=True,
            apply_page_skew=True,
            page_skew_angle=0.9,
        )

        aged_img = augmenter.apply_historical_paper_and_aging(
            page_img,
            add_foxing=True,
            add_fold_crease=True,
            edge_wear=True,
        )

        out_file = output_dir / filename
        aged_img.save(out_file)
        print(f"  [OK] {sys_name.upper():14s} Beispielseite gespeichert: {out_file}")
        generated_files.append(out_file)

    return generated_files


def run_batch_demo():
    print("\n" + "=" * 60)
    print("Starte Batch-Pipeline Test (5 Seiten mit paralleler Erzeugung)...")
    print("=" * 60)

    pipeline = HistoricalBatchPipeline(output_dir="data/synthetic/historical_pretraining")
    summary = pipeline.generate_batch(num_pages=5, train_ratio=0.8, val_ratio=0.1)
    return summary


if __name__ == "__main__":
    generate_single_demo()
    run_batch_demo()
