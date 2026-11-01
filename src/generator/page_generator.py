"""
DEK-Seiten-Generator (Ganze Manuskripte / Stenoblöcke).

Erzeugt vollständige, mehrzeilige Kurzschrift-Seiten (z. B. Stenoblock DIN A5 / DIN A4)
inklusive:
- Zeilenumbruch und Satzspiegel
- Stenoblock-Lineatur (4-Linien-System je Zeile, Randlinien)
- Zeilenweiser Handschrift-Dynamik
- Seitenweiter Scan-Effekte (leichte Drehung/Skew, Vignettierung, Randabfall)
- Export von Ground-Truth Bounding-Boxes je Zeile (für Zeilensegmentierung & HTR)
"""

import math
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from src.dek_rules.parser import DekParser, DekWordToken
from src.generator.stroke_model import StrokeAssembler, ContinuousStroke
from src.generator.renderer import DekRenderer
from src.generator.augmenter import HandwritingAugmenter


class DekPageGenerator:
    """Generiert vollständige Manuskriptseiten in Deutscher Einheitskurzschrift."""

    def __init__(
        self,
        output_dir: str = "data/synthetic/pages",
        page_width: int = 1400,
        page_height: int = 1980,
        margin_left: int = 120,
        margin_right: int = 120,
        margin_top: int = 140,
        margin_bottom: int = 140,
        line_spacing_px: float = 75.0,
        stufe_height_px: float = 22.0,
        base_thickness_px: float = 1.8,
        system: str = "dek",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.width = page_width
        self.height = page_height
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.line_spacing = line_spacing_px
        self.stufe_h = stufe_height_px
        self.system = system.strip().lower().replace("-", "_")

        self.parser = DekParser(system=self.system)
        self.assembler = StrokeAssembler(stufe_height_px=stufe_height_px, base_thickness_px=base_thickness_px)
        self.renderer = DekRenderer()
        self.augmenter = HandwritingAugmenter()

    def generate_page(
        self,
        text: str,
        title: Optional[str] = None,
        has_margin_line: bool = True,
        draw_ruling: bool = True,
        apply_page_skew: bool = True,
        page_skew_angle: float = 0.8,
        bg_color: Tuple[int, int, int] = (252, 250, 244),
    ) -> Tuple[Image.Image, Dict]:
        """
        Erzeugt eine vollständige Seite aus einem langen Text.

        Returns:
            Tuple[Image, Dict]: Das gerenderte Seitenbild und die Zeilen-Metadaten (Bounding-Boxes + Text).
        """
        # 1. Berechne verfügbare Zeilen
        content_width = self.width - self.margin_left - self.margin_right
        content_height = self.height - self.margin_top - self.margin_bottom
        num_lines = int(content_height / self.line_spacing)

        # 2. Text in Token parsen
        tokens = self.parser.parse_text(text)
        if not tokens:
            raise ValueError("Keine gültigen Token im übergebenen Text gefunden.")

        # 3. Grundblatt mit optionaler Stenolineatur zeichnen
        img = Image.new("RGB", (self.width, self.height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Lineatur für alle Zeilen zeichnen
        baselines_y = []
        for i in range(num_lines):
            base_y = self.margin_top + (i + 1) * self.line_spacing
            baselines_y.append(base_y)
            if draw_ruling:
                self._draw_steno_ruling(draw, base_y)

        # Optional: Vertikale Randlinie (typisch für deutsche Stenoblöcke)
        if has_margin_line and draw_ruling:
            margin_x = self.margin_left - 25
            draw.line([(margin_x, 50), (margin_x, self.height - 50)], fill=(225, 175, 175), width=2)

        # 4. Wörter auf Zeilen umbrechen (Line Wrapping)
        lines_data: List[Dict] = []
        current_line_idx = 0
        current_x = float(self.margin_left)
        current_line_strokes: List[ContinuousStroke] = []
        current_line_words: List[str] = []
        word_gap = self.stufe_h * 1.4

        for token in tokens:
            # Berechne Breite des Worts
            word_width = max(sum(g.delta_x for g in token.glyphs) * self.stufe_h, 15.0)

            # Prüfe Zeilenumbruch
            if current_x + word_width > (self.width - self.margin_right):
                # Zeile abschließen
                if current_line_strokes:
                    line_meta = self._render_and_record_line(
                        img=img,
                        strokes=current_line_strokes,
                        words=current_line_words,
                        line_idx=current_line_idx,
                        base_y=baselines_y[current_line_idx],
                    )
                    lines_data.append(line_meta)

                current_line_idx += 1
                if current_line_idx >= num_lines:
                    # Seite voll
                    break

                current_x = float(self.margin_left)
                current_line_strokes = []
                current_line_words = []

            # Wort zur aktuellen Zeile assemblieren
            base_y = baselines_y[current_line_idx]
            word_strokes = self.assembler.assemble_word(token.glyphs, start_x=current_x, base_y=base_y)
            current_line_strokes.extend(word_strokes)
            current_line_words.append(token.raw_word)

            current_x += word_width + word_gap

        # Letzte Zeile abschließen, falls noch Inhalte da sind
        if current_line_strokes and current_line_idx < num_lines:
            line_meta = self._render_and_record_line(
                img=img,
                strokes=current_line_strokes,
                words=current_line_words,
                line_idx=current_line_idx,
                base_y=baselines_y[current_line_idx],
            )
            lines_data.append(line_meta)

        # 5. Seiten-Augmentierung (Papiertextur, Beleuchtungsgradient, Gesamtdrehung)
        final_img = self._apply_page_level_effects(img, apply_page_skew, page_skew_angle)

        page_meta = {
            "page_width": self.width,
            "page_height": self.height,
            "num_lines": len(lines_data),
            "lines": lines_data,
        }

        return final_img, page_meta

    def _draw_steno_ruling(self, draw: ImageDraw.ImageDraw, base_y: float):
        """Zeichnet die systemtypische Lineatur für eine Zeile über die gesamte Seitenbreite."""
        line_color = (215, 225, 238)       # Helles Blau für Hilfslinien
        baseline_color = (185, 202, 222)   # Kräftigere Grundlinie

        if self.system == "gabelsberger":
            # Gabelsberger nutzt klassischerweise eine einfache Grundlinie
            levels = [(0.0, baseline_color, 1)]
        else:
            levels = [
                (2.0, line_color, 1),
                (1.0, line_color, 1),
                (0.0, baseline_color, 1),
                (-1.0, line_color, 1),
            ]

        x_start = self.margin_left - 40
        x_end = self.width - self.margin_right + 40

        for lvl, col, w in levels:
            y = base_y - (lvl * self.stufe_h)
            draw.line([(x_start, y), (x_end, y)], fill=col, width=w)

    def _render_and_record_line(
        self,
        img: Image.Image,
        strokes: List[ContinuousStroke],
        words: List[str],
        line_idx: int,
        base_y: float
    ) -> Dict:
        """Rendert die Striche einer Zeile mit leichter handschriftlicher Dynamik ins Seitenbild."""
        draw = ImageDraw.Draw(img)

        # Leichte Handschrift-Variation auf Zeilenebene
        aug_strokes = self.augmenter.augment_strokes(strokes, base_y=base_y)

        min_x, max_x = float("inf"), 0.0
        min_y, max_y = float("inf"), 0.0

        for stroke in aug_strokes:
            pts = stroke.points
            if len(pts) < 1:
                continue

            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                avg_thick = (p1.thickness + p2.thickness) / 2.0
                r = avg_thick / 2.0

                draw.line([(p1.x, p1.y), (p2.x, p2.y)], fill=(20, 25, 40), width=max(1, int(round(avg_thick))))
                draw.ellipse([p1.x - r, p1.y - r, p1.x + r, p1.y + r], fill=(20, 25, 40))

                min_x = min(min_x, p1.x - r, p2.x - r)
                max_x = max(max_x, p1.x + r, p2.x + r)
                min_y = min(min_y, p1.y - r, p2.y - r)
                max_y = max(max_y, p1.y + r, p2.y + r)

        line_text = " ".join(words)
        bbox = [int(min_x), int(min_y), int(max_x), int(max_y)] if min_x != float("inf") else [0, 0, 0, 0]

        return {
            "line_index": line_idx,
            "baseline_y": int(base_y),
            "text": line_text,
            "bbox": bbox
        }

    def _apply_page_level_effects(
        self,
        img: Image.Image,
        apply_skew: bool,
        max_skew_angle: float
    ) -> Image.Image:
        """Fügt realistische Papiertextur, Scanner-Ausleuchtungsgradienten und Schrägstellung hinzu."""
        arr = np.array(img, dtype=np.float32)

        # 1. Feines Papier-Rauschen
        noise = np.random.normal(0, 3.5, arr.shape)
        arr = np.clip(arr + noise, 0, 255)

        # 2. Scanner-Beleuchtungsgradient (Vignettierung / ungleichmäßiges Licht)
        h, w, _ = arr.shape
        grad_x = np.linspace(0.97, 1.02, w)
        grad_y = np.linspace(0.98, 1.03, h)
        grid = np.outer(grad_y, grad_x)[:, :, np.newaxis]
        arr = np.clip(arr * grid, 0, 255).astype(np.uint8)

        page_img = Image.fromarray(arr)

        # 3. Leichte ganzheitliche Seitendrehung (wie ein schräg eingelegtes Blatt im Scanner)
        if apply_skew:
            angle = random.uniform(-max_skew_angle, max_skew_angle)
            # Mit leichtem Rand drehen
            page_img = page_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(245, 242, 235))

        return page_img
