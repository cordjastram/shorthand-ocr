"""
Post-War Augmenter: Spezialisierte Bild- und Textureffekte für Dokumente der Ära 1945–1955.

Simuliert:
1. Holzschliffpapier (Vergilbung, Randbräunung, ungleichmäßige Fasern)
2. Stockflecken (Foxing / Feuchtigkeitsflecken historischer Dokumente)
3. Faltspuren & Knicklinien (typisch für gefaltete Amts- und Steno-Blätter)
4. Schreibwerkzeuge der 1940er/50er:
   - Weicher Bleistift (Graphitkorn, wechselnder Anpreßdruck, Graphitgrau)
   - Verblasste Eisengallustinte (Sepia-Schwarz, feines Ausbluten in Papierfasern)
"""

import math
import random
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class PostWarAugmenter:
    """Wendet historische Nachkriegseffekte auf DEK-Seiten und Zeilen an."""

    def __init__(
        self,
        paper_tone: str = "aged_yellow",  # "aged_yellow", "wood_pulp_brown", "faded_cream"
        tool_type: str = "pencil",        # "pencil" (Bleistift) oder "faded_ink" (Eisengallustinte)
    ):
        self.paper_tone = paper_tone
        self.tool_type = tool_type

    def get_tool_color_and_thickness(self) -> Tuple[Tuple[int, int, int], float]:
        """Liefert zeittypische Farbtöne für das gewählte Schreibwerkzeug."""
        if self.tool_type == "pencil":
            # Graphit-Grau mit feinen Nuancen
            gray_val = random.randint(45, 65)
            # Minimaler warmer Farbstich vom Papierreflex
            return (gray_val, gray_val, gray_val + random.randint(0, 4)), random.uniform(1.6, 2.2)
        else:
            # Gealterte Eisengallustinte (Sepia / Schwarzbraun)
            r = random.randint(35, 52)
            g = random.randint(28, 42)
            b = random.randint(22, 34)
            return (r, g, b), random.uniform(1.5, 2.0)

    def apply_historical_paper_and_aging(
        self,
        img: Image.Image,
        add_foxing: bool = True,
        add_fold_crease: bool = True,
        edge_wear: bool = True,
    ) -> Image.Image:
        """
        Wandelt ein gerendertes Bild in ein authentisches Nachkriegsdokument um.
        """
        w, h = img.size
        arr = np.array(img, dtype=np.float32)

        # 1. Grund-Papierfarbton bestimmen (Holzschliff, vergilbt oder reinweiß)
        if self.paper_tone == "wood_pulp_brown":
            base_tint = np.array([236.0, 222.0, 196.0], dtype=np.float32)  # Stark vergilbtes Holzschliffpapier
        elif self.paper_tone == "faded_cream":
            base_tint = np.array([246.0, 241.0, 228.0], dtype=np.float32)  # Gebleichtes, leicht vergilbtes Papier
        elif self.paper_tone == "pure_white":
            base_tint = np.array([254.0, 254.0, 254.0], dtype=np.float32)  # Klares weißes Briefpapier
        else:
            base_tint = np.array([242.0, 233.0, 212.0], dtype=np.float32)  # Klassischer Vintage-Ton

        # Mische Original-Hintergrund sanft in den historischen Farbton
        # Maske: Dunkle Striche behalten ihren Kontrast, weißer/heller Hintergrund nimmt Papierfarbe an
        luminance = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114) / 255.0
        luminance = np.clip(luminance, 0.0, 1.0)[:, :, np.newaxis]

        # Wenn Hintergrund weiß war (255), nimm base_tint. Wenn Vordergrund dunkel war, behalte Strichfarbe
        paper_canvas = np.ones((h, w, 3), dtype=np.float32) * base_tint[np.newaxis, np.newaxis, :]
        blended = arr * (1.0 - luminance) + paper_canvas * luminance

        # 2. Randvergilbung & Eckenbräunung (Oxidation über 75+ Jahre)
        if edge_wear:
            # Distanz zum Zentrum
            y_coords, x_coords = np.ogrid[:h, :w]
            norm_x = (x_coords - w / 2.0) / (w / 2.0)
            norm_y = (y_coords - h / 2.0) / (h / 2.0)
            dist_sq = (norm_x ** 2) * 0.6 + (norm_y ** 2) * 0.8
            edge_factor = 1.0 - np.clip(dist_sq * 0.12, 0.0, 0.22)[:, :, np.newaxis]

            # Ränder werden dunkler und etwas gelb-bräunlicher
            blended *= edge_factor

        # 3. Stockflecken (Foxing / Feuchtigkeitsflecken)
        if add_foxing:
            blended = self._add_foxing_spots(blended, w, h)

        # 4. Faltspur (z. B. typische horizontale Faltung des Briefs/Protokolls)
        if add_fold_crease:
            blended = self._add_fold_crease(blended, w, h)

        # 5. Graphit- oder Tinten-Textur (Körnung)
        noise = np.random.normal(0, 3.2, blended.shape)
        blended = np.clip(blended + noise, 0, 255).astype(np.uint8)

        aged_img = Image.fromarray(blended)

        # 6. Sanfter Weichzeichner (0.3 px) für natürliche Tinten-/Graphit-Absorption
        aged_img = aged_img.filter(ImageFilter.GaussianBlur(radius=0.35))

        return aged_img

    def _add_foxing_spots(self, arr: np.ndarray, w: int, h: int, num_spots: Optional[int] = None) -> np.ndarray:
        """Erzeugt unregelmäßige, bräunliche Altersflecken (Stockflecken)."""
        if num_spots is None:
            num_spots = random.randint(4, 12)

        for _ in range(num_spots):
            cx = random.randint(30, w - 30)
            cy = random.randint(30, h - 30)
            radius = random.uniform(15.0, 60.0)
            intensity = random.uniform(0.06, 0.16)

            # Bounding box um den Fleck
            x0 = max(0, int(cx - radius * 2))
            x1 = min(w, int(cx + radius * 2))
            y0 = max(0, int(cy - radius * 2))
            y1 = min(h, int(cy + radius * 2))

            y_grid, x_grid = np.ogrid[y0:y1, x0:x1]
            dist_sq = ((x_grid - cx) / radius) ** 2 + ((y_grid - cy) / (radius * random.uniform(0.7, 1.3))) ** 2
            blob = np.exp(-0.5 * dist_sq)

            # Bräunliche Tönung
            spot_color = np.array([0.72, 0.60, 0.44], dtype=np.float32)
            alpha = (blob * intensity)[:, :, np.newaxis]

            patch = arr[y0:y1, x0:x1]
            arr[y0:y1, x0:x1] = patch * (1.0 - alpha) + (patch * spot_color) * alpha

        return arr

    def _add_fold_crease(self, arr: np.ndarray, w: int, h: int) -> np.ndarray:
        """Erzeugt eine Faltlinie (z. B. mittig horizontal gefalteter Bogen)."""
        # Horizontale Falte etwa im mittleren Drittel
        fold_y = int(h * random.uniform(0.42, 0.58))
        thickness = random.randint(3, 7)

        y_indices = np.arange(max(0, fold_y - thickness), min(h, fold_y + thickness))
        for y in y_indices:
            dist = abs(y - fold_y)
            # Durch Faltung leicht abgedunkelte/geknickte Faserlinie
            shadow = 1.0 - (0.07 * (1.0 - dist / thickness))
            arr[y, :, :] *= shadow

        return arr
