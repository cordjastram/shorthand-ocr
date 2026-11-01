"""
Automatisches Deskewing (Schräglagenkorrektur) für Kurzschrift-Scans.

Ermittelt den Neigungswinkel der Zeilen/Hilfslinien über
Projektionsprofil-Varianzanalyse und richtet das Dokument horizontal aus.
"""

import math
from typing import Tuple
import numpy as np
from PIL import Image


class DocumentDeskewer:
    """Richtet schräg gescannte Kurzschrift-Dokumente automatisch gerade aus."""

    def __init__(self, angle_range: float = 6.0, angle_step: float = 0.25):
        """
        Args:
            angle_range: Maximaler Suchbereich in Grad (z. B. ±6 Grad).
            angle_step: Schrittweite für die Winkelsuche in Grad.
        """
        self.angle_range = angle_range
        self.angle_step = angle_step

    def detect_skew_angle(self, img: Image.Image) -> float:
        """
        Ermittelt den Neigungswinkel über die maximale Varianz des horizontalen Projektionsprofils.
        Wenn Zeilen exakt horizontal liegen, sind Spitzen (Text) und Täler (Zwischenräume) am schärfsten.
        """
        # Auf handhabbare Größe herunterskalieren für schnelle Berechnung
        thumb = img.convert("L")
        w, h = thumb.size
        max_dim = 1000
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            thumb = thumb.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

        # Invertieren (Tinte = hohes Signal)
        arr = 255.0 - np.array(thumb, dtype=np.float32)

        # Ränder ausblenden (nur zentraler Inhalt relevant)
        th_h, th_w = arr.shape
        margin_y = int(th_h * 0.1)
        margin_x = int(th_w * 0.1)
        core = arr[margin_y:th_h - margin_y, margin_x:th_w - margin_x]

        best_angle = 0.0
        max_variance = -1.0

        angles = np.arange(-self.angle_range, self.angle_range + self.angle_step, self.angle_step)

        # Pillow Image des Kernbereichs für schnelle Drehung
        core_img = Image.fromarray(core)

        for angle in angles:
            # Drehe um den Testwinkel
            rotated = core_img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0)
            rot_arr = np.array(rotated, dtype=np.float32)

            # Horizontale Projektion (Summe je Zeile)
            proj = np.sum(rot_arr, axis=1)

            # Varianz des Profils
            var = float(np.var(proj))

            if var > max_variance:
                max_variance = var
                best_angle = float(angle)

        return best_angle

    def deskew_image(self, img: Image.Image) -> Tuple[Image.Image, float]:
        """
        Ermittelt den Winkel und rotiert das Bild gerade.

        Returns:
            Tuple[Image, float]: Begradigtes Bild und der korrigierte Winkel in Grad.
        """
        angle = self.detect_skew_angle(img)

        # Wenn der Winkel minimal ist (< 0.1°), keine Drehung nötig
        if abs(angle) < 0.1:
            return img, 0.0

        # Das Bild mit weißer Hintergrundfüllung drehen
        # Winkel positiv rotiert gegen den Uhrzeigersinn in Pillow
        deskewed = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))
        return deskewed, angle
