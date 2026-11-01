"""
Mathematisches Pfad- und Kurvenmodell für DEK-Striche.

Verbindet einzelne Glyphen zu zusammenhängenden Wörtern und Zeilen,
interpoliert Stützpunkte mit Splines (für natürliche Handschriftkurven)
und verwaltet Stufenlagen sowie Strichstärken.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from src.dek_rules.symbols import DekGlyph, StrokeSegment


@dataclass
class RenderPoint:
    """Punkt eines gezeichneten Strichs."""
    x: float
    y: float
    thickness: float  # Strichstärke an dieser Stelle


@dataclass
class ContinuousStroke:
    """Zusammenhängender Linienzug eines Wortes."""
    points: List[RenderPoint]


class StrokeAssembler:
    """Verbindet Glyphen-Sequenzen zu kontinuierlichen Koordinatenpfaden."""

    def __init__(self, stufe_height_px: float = 30.0, base_thickness_px: float = 2.0):
        """
        Args:
            stufe_height_px: Höhe einer DEK-Stufe in Pixeln.
            base_thickness_px: Basis-Strichbreite (Haarstrich).
        """
        self.stufe_h = stufe_height_px
        self.base_thickness = base_thickness_px

    def assemble_word(self, glyphs: List[DekGlyph], start_x: float = 50.0, base_y: float = 150.0) -> List[ContinuousStroke]:
        """
        Erzeugt aus einer Liste von Glyphen zusammenhängende Pfade.
        
        y-Koordinate: In Bildkoordinaten wächst y nach UNTEN, daher:
        pixel_y = base_y - (stufe_y * stufe_height_px)
        """
        if not glyphs:
            return []

        strokes: List[ContinuousStroke] = []
        cur_x = start_x
        # Grundlinie ist relative Höhe 0.0
        cur_level = 0.0

        for glyph in glyphs:
            for segment in glyph.segments:
                seg_pts: List[RenderPoint] = []
                num_points = len(segment.points)
                if num_points == 0:
                    continue

                # Berechne absolute Punkte
                raw_coords = []
                for dx, dy in segment.points:
                    px = cur_x + dx * self.stufe_h
                    py = base_y - (cur_level + dy) * self.stufe_h
                    raw_coords.append((px, py))

                # Dicke: Haarstrich vs. Schattierung
                thickness = self.base_thickness * segment.thickness_factor

                # Wenn mehr als 2 Punkte da sind, interpoliere mit Splines für weiche Kurven
                if len(raw_coords) >= 4:
                    smooth_coords = self._smooth_curve(raw_coords)
                    for x, y in smooth_coords:
                        seg_pts.append(RenderPoint(x=x, y=y, thickness=thickness))
                elif len(raw_coords) > 1:
                    # Lineare Unterteilung für gleichmäßigen Strichauftrag
                    interp_coords = self._subdivide_line(raw_coords)
                    for x, y in interp_coords:
                        seg_pts.append(RenderPoint(x=x, y=y, thickness=thickness))
                else:
                    px, py = raw_coords[0]
                    seg_pts.append(RenderPoint(x=px, y=py, thickness=thickness))

                strokes.append(ContinuousStroke(points=seg_pts))

            # Vorschub zum nächsten Zeichen
            cur_x += glyph.delta_x * self.stufe_h
            cur_level += glyph.delta_y

        return strokes

    def _subdivide_line(self, coords: List[Tuple[float, float]], step: float = 2.0) -> List[Tuple[float, float]]:
        """Unterteilt Streckenabschnitte fein, damit der Strichzeichner keine Lücken hat."""
        dense = []
        for i in range(len(coords) - 1):
            p1 = np.array(coords[i])
            p2 = np.array(coords[i + 1])
            dist = np.linalg.norm(p2 - p1)
            num_steps = max(int(dist / step), 1)
            for s in range(num_steps):
                t = s / num_steps
                pt = p1 * (1 - t) + p2 * t
                dense.append((float(pt[0]), float(pt[1])))
        dense.append(coords[-1])
        return dense

    def _smooth_curve(self, coords: List[Tuple[float, float]], samples_per_segment: int = 15) -> List[Tuple[float, float]]:
        """Catmull-Rom-Spline-Interpolation in reinem NumPy für natürliche Handschriftkurven."""
        if len(coords) < 3:
            return coords

        pts = np.array(coords, dtype=np.float32)

        # Virtuelle Endpunkte extrapolieren für fließenden Start und Auslauf
        p_start = 2 * pts[0] - pts[1]
        p_end = 2 * pts[-1] - pts[-2]
        extended = np.vstack([[p_start], pts, [p_end]])

        smooth_points: List[Tuple[float, float]] = []

        for i in range(len(pts) - 1):
            p0 = extended[i]
            p1 = extended[i + 1]
            p2 = extended[i + 2]
            p3 = extended[i + 3]

            t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=(i == len(pts) - 2))
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom Matrixformel: 0.5 * ( (2*P1) + (-P0+P2)*t + (2*P0 - 5*P1 + 4*P2 - P3)*t^2 + (-P0 + 3*P1 - 3*P2 + P3)*t^3 )
            a0 = 2.0 * p1
            a1 = -p0 + p2
            a2 = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
            a3 = -p0 + 3.0 * p1 - 3.0 * p2 + p3

            curve_pts = 0.5 * (
                a0[np.newaxis, :] +
                np.outer(t, a1) +
                np.outer(t2, a2) +
                np.outer(t3, a3)
            )

            for cp in curve_pts:
                smooth_points.append((float(cp[0]), float(cp[1])))

        return smooth_points
