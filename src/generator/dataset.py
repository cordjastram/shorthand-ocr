"""
Synthetischer DEK-Datensatz-Generator.

Kombiniert Parser, Assembler, Augmentierung und Renderer,
um aus Textkorpora automatisch beschriftete Trainingsdaten für
das HTR-Modell zu generieren.
"""

import os
import json
import csv
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from src.dek_rules.parser import DekParser, DekWordToken
from src.generator.stroke_model import StrokeAssembler, ContinuousStroke
from src.generator.renderer import DekRenderer
from src.generator.augmenter import HandwritingAugmenter


class DekDatasetGenerator:
    """Erzeugt synthetische DEK-Trainingsbeispiele."""

    def __init__(
        self,
        output_dir: str = "data/synthetic",
        stufe_height_px: float = 32.0,
        base_thickness_px: float = 2.0,
        system: str = "dek",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)

        self.system = system.strip().lower().replace("-", "_")
        self.parser = DekParser(system=self.system)
        self.assembler = StrokeAssembler(stufe_height_px=stufe_height_px, base_thickness_px=base_thickness_px)
        self.renderer = DekRenderer()
        self.augmenter = HandwritingAugmenter()
        self.stufe_h = stufe_height_px

    def generate_single(
        self,
        text: str,
        apply_augmentation: bool = True,
        draw_guidelines: bool = True,
        width: int = 700,
        height: int = 240,
        base_y: float = 140.0,
    ) -> Optional[Image.Image]:
        """Erzeugt ein Bild für eine einzelne Textzeile oder ein Wort."""
        tokens = self.parser.parse_text(text)
        if not tokens:
            return None

        # Sammle alle Glyphen über die Wörter mit Wortabständen
        all_strokes: List[ContinuousStroke] = []
        cur_x = 40.0
        word_spacing = self.stufe_h * 1.5

        for token in tokens:
            word_strokes = self.assembler.assemble_word(
                glyphs=token.glyphs,
                start_x=cur_x,
                base_y=base_y
            )
            all_strokes.extend(word_strokes)

            # Berechne Vorschub für das nächste Wort
            word_width = sum(g.delta_x for g in token.glyphs) * self.stufe_h
            cur_x += max(word_width, 20.0) + word_spacing

        if not all_strokes:
            return None

        # Berechne benötigte Bildbreite dynamisch
        img_width = max(width, int(cur_x + 50.0))

        # Augmentierung auf Strichebene
        if apply_augmentation:
            strokes_to_draw = self.augmenter.augment_strokes(all_strokes, base_y=base_y)
        else:
            strokes_to_draw = all_strokes

        # Rendern
        img = self.renderer.render_to_image(
            strokes=strokes_to_draw,
            width=img_width,
            height=height,
            draw_guidelines=draw_guidelines,
            base_y=base_y,
            stufe_h=self.stufe_h,
        )

        # Augmentierung auf Bildebene (Papier, Rauschen)
        if apply_augmentation:
            img = self.augmenter.augment_image(img)

        return img

    def generate_dataset(
        self,
        texts: List[str],
        samples_per_text: int = 1,
        manifest_filename: str = "dataset.csv"
    ) -> List[Dict]:
        """Erzeugt einen vollständigen Datensatz mit Manifest-Datei."""
        manifest = []
        sample_id = 0

        manifest_path = self.output_dir / manifest_filename
        fieldnames = ["id", "image_path", "text", "is_augmented"]

        with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for text in tqdm(texts, desc="Generiere DEK-Datensatz"):
                for s_idx in range(samples_per_text):
                    img_name = f"dek_sample_{sample_id:06d}.png"
                    img_path = self.output_dir / "images" / img_name

                    # Erstes Exemplar ideal (clean), weitere augmentiert
                    is_aug = (s_idx > 0)
                    img = self.generate_single(text, apply_augmentation=is_aug)

                    if img:
                        img.save(img_path)
                        record = {
                            "id": sample_id,
                            "image_path": str(img_path.as_posix()),
                            "text": text,
                            "is_augmented": is_aug
                        }
                        writer.writerow(record)
                        manifest.append(record)
                        sample_id += 1

        print(f"Datensatz erfolgreich generiert: {len(manifest)} Bilder in {self.output_dir}")
        return manifest
