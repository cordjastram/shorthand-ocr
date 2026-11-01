"""
Handschrift- und Scan-Augmentierung für synthetische DEK-Daten.

Fügt realistische Handschrift- und Scaneigenschaften hinzu:
- Handschrift-Schräglage (Slant / Shear)
- Strichzitter (Mikro-Jitter durch Handbewegung)
- Papiertextur und Scanrauschen
- Leicht ungleichmäßige Helligkeitsverläufe (Scannerbeleuchtung)
"""

import random
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageFilter
from src.generator.stroke_model import ContinuousStroke, RenderPoint


class HandwritingAugmenter:
    """Verzerrt und variiert ideale Vektorstriche für realistische Handschrift."""

    def __init__(
        self,
        slant_range: Tuple[float, float] = (-0.15, 0.25),   # Scherung (Handschrift-Neigung)
        jitter_std: float = 0.6,                           # Standardabweichung des Strichzitterns
        scale_range: Tuple[float, float] = (0.9, 1.1),     # Leichte Größenschwankungen
        thickness_variation: float = 0.2,                  # Schwankung der Strichstärke
    ):
        self.slant_range = slant_range
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.thickness_variation = thickness_variation

    def augment_strokes(self, strokes: List[ContinuousStroke], base_y: float = 150.0) -> List[ContinuousStroke]:
        """Wendet geometrische Handschrift-Variationen auf Vektorstriche an."""
        slant = random.uniform(*self.slant_range)
        scale_x = random.uniform(*self.scale_range)
        scale_y = random.uniform(*self.scale_range)
        thick_mult = random.uniform(1.0 - self.thickness_variation, 1.0 + self.thickness_variation)

        augmented_strokes = []
        for stroke in strokes:
            new_pts = []
            for pt in stroke.points:
                # 1. Skalierung relativ zur Grundlinie
                dx = (pt.x) * scale_x
                dy = (pt.y - base_y) * scale_y

                # 2. Neigung (Slant): Horizontale Verschiebung proportional zur Höhe
                # Punkte oberhalb der Grundlinie (dy < 0 in Pixeln) neigen sich nach rechts
                slanted_x = dx - dy * slant
                slanted_y = base_y + dy

                # 3. Zitter (Jitter)
                jitter_x = random.gauss(0, self.jitter_std)
                jitter_y = random.gauss(0, self.jitter_std)

                # 4. Dickenschwankung
                pt_thick = max(1.0, pt.thickness * thick_mult + random.gauss(0, 0.1))

                new_pts.append(
                    RenderPoint(
                        x=slanted_x + jitter_x,
                        y=slanted_y + jitter_y,
                        thickness=pt_thick
                    )
                )
            augmented_strokes.append(ContinuousStroke(points=new_pts))

        return augmented_strokes

    def augment_image(self, img: Image.Image) -> Image.Image:
        """Fügt dem gerenderten Bild Papiertextur, Kontrastrauschen und Weichzeichnung hinzu."""
        arr = np.array(img, dtype=np.float32)

        # 1. Papier-Korn / Gauss-Rauschen
        noise = np.random.normal(0, 4.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255)

        # 2. Leichter Scanner-Helligkeitsverlauf (Vignette / ungleichmäßige Ausleuchtung)
        h, w, _ = arr.shape
        gradient_x = np.linspace(1.0, random.uniform(0.95, 1.05), w)
        gradient_y = np.linspace(1.0, random.uniform(0.96, 1.04), h)
        grid = np.outer(gradient_y, gradient_x)[:, :, np.newaxis]
        arr = np.clip(arr * grid, 0, 255).astype(np.uint8)

        aug_img = Image.fromarray(arr)

        # 3. Minimale Unschärfe für natürlichen Tintenfluss (Anti-Aliasing)
        if random.random() < 0.5:
            aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=0.4))

        return aug_img
