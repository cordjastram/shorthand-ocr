"""
Zeilensegmentierung für Kurzschrift-Manuskripte.

Zerlegt eine begradigte Seite in einzelne Zeilenstreifen:
1. Horizontale Projektionsprofil-Analyse mit Gauß-Glättung
2. Robuste Peak- und Tal-Erkennung (funktioniert auch bei linierem Papier)
3. Schnittstellenbestimmung in den lokalen Tälern zwischen Textzeilen
4. Horizontales Trimmen der Ränder (Bounding Box)
5. Sicherheits-Padding für DEK-Ober- und Unterlängen
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image


@dataclass
class SegmentedLine:
    """Repräsentiert eine aus der Seite extrahierte Zeile."""
    line_index: int
    image: Image.Image
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    baseline_y: int


class LineSegmenter:
    """Segmentiert eine begradigte Kurzschriftseite in einzelne Zeilen."""

    def __init__(
        self,
        min_line_distance: int = 40,
        padding_y: int = 14,
        padding_x: int = 15,
        edge_margin_ratio: float = 0.05,
    ):
        self.min_line_distance = min_line_distance
        self.padding_y = padding_y
        self.padding_x = padding_x
        self.edge_margin_ratio = edge_margin_ratio

    def segment_lines(self, img: Image.Image) -> List[SegmentedLine]:
        """Extrahiert alle Textzeilen aus dem übergebenen Bild (mit adaptiver Skalierung für verschiedene Scan-Auflösungen)."""
        gray = img.convert("L")
        w, h = gray.size

        # Adaptive Skalierung relativ zur Referenzhöhe (1700 px)
        scale_factor = max(0.35, min(1.5, h / 1700.0))
        eff_min_dist = max(18, int(round(self.min_line_distance * scale_factor)))
        eff_pad_y = max(4, int(round(self.padding_y * scale_factor)))
        eff_pad_x = max(6, int(round(self.padding_x * (w / 1200.0))))

        # Invertieren: Dunkle Tinte/Bleistift = hohes Signal
        arr = 255.0 - np.array(gray, dtype=np.float32)

        # Schwache Hintergrund-Schwellenwertfilterung (unterdrückt Papiergrau bei der Projektion)
        signal_arr = np.where(arr > 25.0, arr, 0.0)

        # 1. Horizontale Projektion (Summe entlang der x-Achse)
        margin_x = int(w * 0.05)
        core_arr = signal_arr[:, margin_x:max(margin_x + 10, w - margin_x)]
        proj = np.sum(core_arr, axis=1)

        # 2. Adaptive 1D Gauß-Glättung des Profils
        sigma = max(2.5, 7.0 * scale_factor)
        kernel_size = int(round(sigma * 5)) | 1  # ungerade Zahl erzwingen
        x_grid = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)
        gauss_kernel = np.exp(-0.5 * (x_grid / sigma) ** 2)
        gauss_kernel /= np.sum(gauss_kernel)
        smooth = np.convolve(proj, gauss_kernel, mode="same")

        # 3. Peak-Erkennung (Zeilenzentren)
        min_y = int(h * max(0.01, self.edge_margin_ratio * 0.4))
        max_y = int(h * (1.0 - max(0.01, self.edge_margin_ratio * 0.4)))

        # Dynamischer Schwellenwert mit lokalem gleitendem Mittelwert, damit auch kurze Zeilen
        # (Datum, Anrede, Grußformel) zuverlässig als Peaks erkannt werden
        global_mean = float(np.mean(smooth[min_y:max_y]))
        thresh = global_mean * 0.35

        # Lokale Maxima identifizieren
        candidates = []
        for y in range(min_y + 1, max_y - 1):
            if smooth[y] > smooth[y - 1] and smooth[y] >= smooth[y + 1] and smooth[y] > thresh:
                candidates.append(y)

        # Mindestabstand zwischen Zeilen durchsetzen (höchster Peak gewinnt)
        peaks: List[int] = []
        for p in candidates:
            if not peaks or (p - peaks[-1] >= eff_min_dist):
                peaks.append(p)
            elif smooth[p] > smooth[peaks[-1]]:
                peaks[-1] = p

        if not peaks:
            return []

        # 4. Schnittpunkte (Täler) zwischen den Peaks bestimmen
        valleys: List[int] = []
        top_cut = max(0, peaks[0] - int(eff_min_dist * 0.75))
        valleys.append(top_cut)

        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            valley_y = p1 + int(np.argmin(smooth[p1:p2]))
            valleys.append(valley_y)

        bottom_cut = min(h, peaks[-1] + int(eff_min_dist * 0.75))
        valleys.append(bottom_cut)

        # 5. Zeilen-Crops ausschneiden und horizontales Bounding-Box-Trimmen durchführen
        segmented_lines: List[SegmentedLine] = []

        for idx in range(len(peaks)):
            y0 = valleys[idx]
            y1 = valleys[idx + 1]
            peak_y = peaks[idx]

            # Vertikales Padding
            padded_y0 = max(0, y0 - eff_pad_y)
            padded_y1 = min(h, y1 + eff_pad_y)

            # Horizontale Bounding Box bestimmen
            line_slice = signal_arr[padded_y0:padded_y1, :]
            col_sums = np.sum(line_slice, axis=0)

            col_thresh = max(30.0, float(np.mean(col_sums)) * 0.18)
            active_cols = np.where(col_sums > col_thresh)[0]

            if len(active_cols) < max(8, int(20 * scale_factor)):
                continue

            x0 = max(0, int(active_cols[0]) - eff_pad_x)
            x1 = min(w, int(active_cols[-1]) + eff_pad_x)

            if (x1 - x0) < max(25, int(50 * scale_factor)):
                continue

            # Prüfe, ob tatsächlicher Text vorhanden ist (nicht nur leere Lineatur)
            crop_content = arr[padded_y0:padded_y1, x0:x1]
            ink_pixels = int(np.sum(crop_content > 55.0))
            min_ink_required = max(60, int(300 * (scale_factor ** 2)))
            if ink_pixels < min_ink_required:
                continue

            # Zeilenbild zuschneiden
            line_crop = img.crop((x0, padded_y0, x1, padded_y1))

            segmented_lines.append(
                SegmentedLine(
                    line_index=len(segmented_lines),
                    image=line_crop,
                    bbox=(x0, padded_y0, x1, padded_y1),
                    baseline_y=peak_y
                )
            )

        return segmented_lines
