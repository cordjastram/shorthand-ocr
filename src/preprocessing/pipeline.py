"""
Master-Pipeline für die Scan-Vorverarbeitung von Kurzschrift-Dokumenten.

Workflow:
1. Schräglagenkorrektur (Deskewing)
2. Kontrast- und Hintergrundoptimierung (CLAHE)
3. Automatische Zeilensegmentierung
4. Speicherung der Zeilen-Crops und eines visuellen Prüfbildes (Debug-Overlay)
5. Export eines JSON-Manifests für nachgelagerte HTR
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image, ImageDraw

from src.preprocessing.deskew import DocumentDeskewer
from src.preprocessing.enhancer import DocumentEnhancer
from src.preprocessing.segmenter import LineSegmenter, SegmentedLine


class ScanPreprocessingPipeline:
    """Verarbeitet rohe Scans von Kurzschrift-Manuskripten."""

    def __init__(
        self,
        output_dir: str = "data/processed",
        deskew_range: float = 6.0,
        clahe_clip_limit: float = 2.0,
    ):
        self.output_dir = Path(output_dir)
        self.lines_dir = self.output_dir / "lines"
        self.debug_dir = self.output_dir / "debug"

        self.lines_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.deskewer = DocumentDeskewer(angle_range=deskew_range)
        self.enhancer = DocumentEnhancer(clahe_clip_limit=clahe_clip_limit)
        self.segmenter = LineSegmenter()

    def process_document(
        self,
        image_path: str,
        document_id: Optional[str] = None,
        save_debug_viz: bool = True,
    ) -> Dict:
        """
        Verarbeitet eine gescannte Seite vollständig.

        Args:
            image_path: Pfad zum gescannten Bild (TIFF/PNG/JPEG).
            document_id: Optionaler Bezeichner (Standard: Dateiname ohne Endung).
            save_debug_viz: Ob ein Kontrollbild mit eingezeichneten Zeilenboxen gespeichert werden soll.

        Returns:
            Dict: Metadaten der extrahierten Zeilen und Pfade.
        """
        input_file = Path(image_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Scan-Datei nicht gefunden: {image_path}")

        doc_id = document_id or input_file.stem
        raw_img = Image.open(input_file).convert("RGB")

        # 1. Deskewing
        deskewed_img, angle = self.deskewer.deskew_image(raw_img)

        # 2. Kontrastoptimierung
        enhanced_img = self.enhancer.enhance(deskewed_img)

        # 3. Zeilensegmentierung
        segmented_lines = self.segmenter.segment_lines(enhanced_img)

        # 4. Zeilen-Crops speichern
        line_records = []
        for line in segmented_lines:
            line_filename = f"{doc_id}_line_{line.line_index:03d}.png"
            line_path = self.lines_dir / line_filename
            line.image.save(line_path)

            line_records.append({
                "line_index": line.line_index,
                "line_id": f"{doc_id}_{line.line_index:03d}",
                "image_path": str(line_path.as_posix()),
                "bbox": list(line.bbox),
                "baseline_y": line.baseline_y,
                "width": line.image.width,
                "height": line.image.height,
            })

        # 5. Kontrollbild mit Bounding Boxes speichern
        debug_viz_path = None
        if save_debug_viz:
            debug_viz = deskewed_img.copy()
            draw = ImageDraw.Draw(debug_viz)

            # Farben für Boxen
            box_color = (0, 160, 80)
            baseline_color = (220, 50, 50)

            for line in segmented_lines:
                x0, y0, x1, y1 = line.bbox
                # Bounding Box zeichnen
                draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)
                # Baseline andeuten
                draw.line([(x0, line.baseline_y), (x1, line.baseline_y)], fill=baseline_color, width=1)
                # Zeilennummer beschriften
                draw.text((x0 + 4, y0 + 2), f"L{line.line_index + 1}", fill=(0, 120, 50))

            debug_filename = f"{doc_id}_segmented_debug.png"
            debug_viz_path = self.debug_dir / debug_filename
            debug_viz.save(debug_viz_path)

        # 6. Dokument-Manifest speichern
        result = {
            "document_id": doc_id,
            "source_image": str(input_file.as_posix()),
            "corrected_angle_deg": round(angle, 2),
            "num_lines_extracted": len(line_records),
            "debug_image": str(debug_viz_path.as_posix()) if debug_viz_path else None,
            "lines": line_records,
        }

        manifest_path = self.output_dir / f"{doc_id}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        result["manifest_path"] = str(manifest_path.as_posix())
        return result
