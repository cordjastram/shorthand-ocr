"""
DEK-Renderer.

Rendert interpolierte DEK-Strichpfade in:
1. Rasterbilder (Pillow Image mit variabler Strichstärke und Kantenglättung)
2. Vektorformate (SVG-Markup)
"""

from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
import numpy as np
from src.generator.stroke_model import ContinuousStroke, RenderPoint


class DekRenderer:
    """Zeichnet DEK-Striche auf ein Bild oder erzeugt SVG."""

    def __init__(
        self,
        ink_color: Tuple[int, int, int] = (25, 30, 45),      # Dunkles Stenotinten-Blau / Schwarz
        bg_color: Tuple[int, int, int] = (252, 250, 245),    # Leicht getöntes Papier
    ):
        self.ink_color = ink_color
        self.bg_color = bg_color

    def render_to_image(
        self,
        strokes: List[ContinuousStroke],
        width: int = 800,
        height: int = 250,
        draw_guidelines: bool = True,
        base_y: float = 150.0,
        stufe_h: float = 30.0,
    ) -> Image.Image:
        """Rendert die Striche auf ein RGB-Bild mit variabler Strichstärke."""
        img = Image.new("RGB", (width, height), color=self.bg_color)
        draw = ImageDraw.Draw(img)

        # 1. Hilfslinien der DEK zeichnen
        if draw_guidelines:
            self._draw_dek_guidelines(draw, width, base_y, stufe_h)

        # 2. Striche zeichnen (Variables Brush-Modeling für Schattierung)
        for stroke in strokes:
            pts = stroke.points
            if len(pts) < 1:
                continue

            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i + 1]
                avg_thick = (p1.thickness + p2.thickness) / 2.0
                r = avg_thick / 2.0

                # Zeichne Linie mit Endkappen
                draw.line([(p1.x, p1.y), (p2.x, p2.y)], fill=self.ink_color, width=max(1, int(round(avg_thick))))
                draw.ellipse([p1.x - r, p1.y - r, p1.x + r, p1.y + r], fill=self.ink_color)

            # Letzten Endpunkt abrunden
            last = pts[-1]
            r = last.thickness / 2.0
            draw.ellipse([last.x - r, last.y - r, last.x + r, last.y + r], fill=self.ink_color)

        return img

    def _draw_dek_guidelines(self, draw: ImageDraw.ImageDraw, width: int, base_y: float, stufe_h: float):
        """
        Zeichnet das 4-Linien-System der DEK:
        - Oberlinie (+2 Stufen)
        - Obergrenze Grundraum (+1 Stufe)
        - Grundlinie (0 Stufen) -> etwas kräftiger
        - Unterlinie (-1 Stufe)
        """
        line_color = (210, 220, 235)       # Sanftes Linienblau
        baseline_color = (190, 205, 225)   # Grundlinie etwas deutlicher

        levels = [
            (2.0, line_color, 1),      # 2. Stufe oben
            (1.0, line_color, 1),      # 1. Stufe oben
            (0.0, baseline_color, 1),  # Grundlinie
            (-1.0, line_color, 1),     # Unterlinie
        ]

        for lvl, col, w in levels:
            y = base_y - (lvl * stufe_h)
            draw.line([(0, y), (width, y)], fill=col, width=w)

    def render_to_svg(
        self,
        strokes: List[ContinuousStroke],
        width: int = 800,
        height: int = 250,
        draw_guidelines: bool = True,
        base_y: float = 150.0,
        stufe_h: float = 30.0,
    ) -> str:
        """Erzeugt SVG-Vektorcode für die Striche."""
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'  <rect width="100%" height="100%" fill="rgb{self.bg_color}" />'
        ]

        if draw_guidelines:
            levels = [(2.0, "#d2dceb"), (1.0, "#d2dceb"), (0.0, "#becde1"), (-1.0, "#d2dceb")]
            for lvl, col in levels:
                y = base_y - (lvl * stufe_h)
                svg_parts.append(f'  <line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="{col}" stroke-width="1" />')

        ink_hex = f"#{self.ink_color[0]:02x}{self.ink_color[1]:02x}{self.ink_color[2]:02x}"
        
        for stroke in strokes:
            pts = stroke.points
            if len(pts) < 2:
                continue

            path_data = [f"M {pts[0].x:.2f} {pts[0].y:.2f}"]
            for p in pts[1:]:
                path_data.append(f"L {p.x:.2f} {p.y:.2f}")

            # Durchschnittliche Dicke
            avg_thick = sum(p.thickness for p in pts) / len(pts)
            svg_parts.append(
                f'  <path d="{" ".join(path_data)}" fill="none" stroke="{ink_hex}" '
                f'stroke-width="{avg_thick:.2f}" stroke-linecap="round" stroke-linejoin="round" />'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
