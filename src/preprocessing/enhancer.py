"""
Bildverbesserung und Kontrastoptimierung für historische Kurzschrift-Scans.

Funktionen:
1. Hintergrund-Ausleuchtungskorrektur (entfernt Schatten von Buchfalzen & Scannerrändern)
2. CLAHE (Contrast Limited Adaptive Histogram Equalization) zur Verstärkung schwacher Bleistiftstriche
3. Subtile Rauschminderung bei Erhalt feinster Haarstriche
"""

import cv2
import numpy as np
from PIL import Image


class DocumentEnhancer:
    """Optimiert Kontrast und Lesbarkeit historischer Kurzschrift-Dokumente."""

    def __init__(self, clahe_clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

    def enhance(self, img: Image.Image) -> Image.Image:
        """
        Wendet adaptive Kontrastverstärkung und Hintergrundbereinigung an.
        
        Wichtig für DEK: Keine reine Schwarz-Weiß-Binarisierung, damit die
        Druckstärken (Schattierung vs. Haarstrich) erhalten bleiben.
        """
        # Konvertiere in OpenCV Graustufen-Array
        arr = np.array(img.convert("L"), dtype=np.uint8)

        # 1. Hintergrund-Ausleuchtung schätzen (morphologisches Closing mit großem Kernel)
        # Glättet Text weg und behält nur die ungleichmäßige Papierhelligkeit
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
        background = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)

        # 2. Ausleuchtungs-Normalisierung (Division durch den Hintergrund)
        # Verhindert, dass Ränder dunkler sind als die Seitenmitte
        norm = np.clip((arr.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)) * 255.0, 0, 255).astype(np.uint8)

        # 3. CLAHE für lokalen Kontrast
        enhanced = self.clahe.apply(norm)

        # 4. Zurück zu PIL Image
        return Image.fromarray(enhanced)
