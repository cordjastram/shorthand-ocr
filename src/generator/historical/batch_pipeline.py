"""
High-Throughput Batch-Pipeline für historisches Pretraining (1945–1955).

Erzeugt parallellisiert auf mehreren CPU-Kernen:
1. Vollständige historische Stenoseiten (DIN A5 / Stenoblöcke)
2. Isolierte Zeilen-Crops für direktes TrOCR / CRNN Pretraining
3. Automatischer Train / Validation / Test Split (z. B. 80 / 10 / 10)
4. Standard-Manifest-Dateien (CSV & JSON Lines) für Hugging Face & PyTorch DataLoader
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

from src.generator.page_generator import DekPageGenerator
from src.generator.historical.corpus_loader import HistoricalCorpusLoader
from src.generator.historical.post_war_augmenter import PostWarAugmenter


def _worker_generate_single_page(args: Tuple) -> Dict:
    """Worker-Funktion für parallele Seitenerzeugung."""
    if len(args) == 7:
        page_id, output_dir_str, text, tool_type, paper_tone, base_seed, system = args
    else:
        page_id, output_dir_str, text, tool_type, paper_tone, base_seed = args
        system = "dek"
    random.seed(base_seed + page_id)

    output_dir = Path(output_dir_str)
    pages_dir = output_dir / "pages"
    lines_dir = output_dir / "lines"

    augmenter = PostWarAugmenter(paper_tone=paper_tone, tool_type=tool_type)
    tool_color, thickness = augmenter.get_tool_color_and_thickness()

    generator = DekPageGenerator(
        output_dir=str(pages_dir),
        page_width=1200,
        page_height=1700,
        margin_left=110,
        margin_right=110,
        margin_top=130,
        margin_bottom=130,
        line_spacing_px=75.0,
        stufe_height_px=22.0,
        base_thickness_px=thickness,
        system=system,
    )
    generator.renderer.ink_color = tool_color

    # 50 % der Seiten ohne Hilfslinien (wie echte Briefe auf Blanko-Papier)
    is_pure_white = (paper_tone == "pure_white")
    draw_ruling = False if is_pure_white else (random.random() < 0.5)
    bg_color = (255, 255, 255) if is_pure_white else (252, 250, 244)

    # Generiere Seite
    page_img, meta = generator.generate_page(
        text=text,
        has_margin_line=draw_ruling,
        draw_ruling=draw_ruling,
        apply_page_skew=True,
        page_skew_angle=random.uniform(0.3, 1.2),
        bg_color=bg_color,
    )

    # Historische Alterung auf die fertige Seite anwenden (auf pure_white ohne Stockflecken/Falten)
    aged_page_img = augmenter.apply_historical_paper_and_aging(
        page_img,
        add_foxing=(not is_pure_white),
        add_fold_crease=(not is_pure_white),
        edge_wear=(not is_pure_white),
    )

    page_filename = f"hist_page_{page_id:06d}.png"
    page_path = pages_dir / page_filename
    aged_page_img.save(page_path)

    # Zeilen-Crops ausschneiden und für HTR vorbereiten
    line_records = []
    w, h = aged_page_img.size

    for line_info in meta.get("lines", []):
        bbox = line_info.get("bbox", [0, 0, 0, 0])
        x0, y0, x1, y1 = bbox

        # Sicherheitsabstand (Padding) um die Zeile
        pad_x = 15
        pad_y = 10
        crop_box = (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(w, x1 + pad_x),
            min(h, y1 + pad_y),
        )

        # Prüfe minimale Zeilengröße
        if (crop_box[2] - crop_box[0] > 40) and (crop_box[3] - crop_box[1] > 20):
            line_img = aged_page_img.crop(crop_box)
            line_filename = f"line_{page_id:06d}_{line_info['line_index']:02d}.png"
            line_path = lines_dir / line_filename
            line_img.save(line_path)

            line_records.append({
                "page_id": page_id,
                "line_id": f"{page_id:06d}_{line_info['line_index']:02d}",
                "line_image_path": str(line_path.as_posix()),
                "text": line_info["text"],
                "width": line_img.width,
                "height": line_img.height,
            })

    return {
        "page_id": page_id,
        "page_image_path": str(page_path.as_posix()),
        "meta": meta,
        "lines": line_records,
    }


class HistoricalBatchPipeline:
    """Orchestriert die Massengenerierung historischer Trainingsdaten."""

    def __init__(self, output_dir: str = "data/synthetic/historical_pretraining", system: str = "dek"):
        self.output_dir = Path(output_dir)
        self.system = system.strip().lower().replace("-", "_")
        self.pages_dir = self.output_dir / "pages"
        self.lines_dir = self.output_dir / "lines"

        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.lines_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_loader = HistoricalCorpusLoader()

    def generate_batch(
        self,
        num_pages: int = 20,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        max_workers: Optional[int] = None,
    ) -> Dict:
        """
        Generiert parallel ein Kontingent an Seiten und extrahiert Zeilen-Trainingsdaten.

        Args:
            num_pages: Anzahl zu generierender Seiten.
            train_ratio: Anteil der Trainingsdaten (z. B. 0.8).
            val_ratio: Anteil der Validierungsdaten (z. B. 0.1, Rest Test).
            max_workers: Maximale CPU-Kerne für parallele Ausführung.
        """
        print(f"\n[1/3] Bereite Generierung von {num_pages} historischen Seiten ({self.system.upper()}) vor...")
        paper_tones = ["aged_yellow", "wood_pulp_brown", "faded_cream", "pure_white", "pure_white"]
        tools = ["pencil", "pencil", "faded_ink"]  # 2:1 Verhältnis Bleistift zu Tinte

        tasks = []
        for pid in range(num_pages):
            text = self.corpus_loader.generate_paragraph(min_sentences=7, max_sentences=12)
            pt = random.choice(paper_tones)
            tool = random.choice(tools)
            tasks.append((pid, str(self.output_dir), text, tool, pt, random.randint(1, 999999), self.system))

        print(f"[2/3] Generiere Seiten parallel...")
        all_lines = []
        all_pages = []

        # Parallele Abarbeitung
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker_generate_single_page, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Erzeuge Seiten"):
                res = future.result()
                all_pages.append(res)
                all_lines.extend(res["lines"])

        print(f"\n[3/3] Erstelle Train/Val/Test Splits & Manifeste für {len(all_lines)} Zeilen...")
        random.shuffle(all_lines)

        n_total = len(all_lines)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        for i, rec in enumerate(all_lines):
            if i < n_train:
                rec["split"] = "train"
            elif i < n_train + n_val:
                rec["split"] = "val"
            else:
                rec["split"] = "test"

        # CSV-Manifest für HTR-Training schreiben
        csv_path = self.output_dir / "lines_manifest.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["line_id", "split", "line_image_path", "text", "width", "height", "page_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_lines:
                writer.writerow(r)

        # JSON-Übersicht für das gesamte Pretraining
        summary = {
            "num_pages": len(all_pages),
            "num_lines": len(all_lines),
            "splits": {
                "train": n_train,
                "val": n_val,
                "test": n_total - n_train - n_val,
            },
            "manifest_csv": str(csv_path.as_posix()),
            "output_directory": str(self.output_dir.as_posix()),
        }

        with open(self.output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("Batch-Generierung erfolgreich abgeschlossen!")
        print(f"  • Gesamtseiten: {summary['num_pages']}")
        print(f"  • Extrahierte Zeilen: {summary['num_lines']}")
        print(f"  • Train: {summary['splits']['train']} | Val: {summary['splits']['val']} | Test: {summary['splits']['test']}")
        print(f"  • Manifest: {csv_path}")
        print("=" * 60)

        return summary
