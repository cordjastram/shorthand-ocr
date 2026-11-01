"""
Automatisierte Tests für die Scan-Vorverarbeitung (Deskewing, Enhancer, Segmenter).
"""

import unittest
from pathlib import Path
from PIL import Image
import numpy as np

from src.preprocessing.deskew import DocumentDeskewer
from src.preprocessing.enhancer import DocumentEnhancer
from src.preprocessing.segmenter import LineSegmenter
from src.preprocessing.pipeline import ScanPreprocessingPipeline


class TestScanPreprocessing(unittest.TestCase):

    def setUp(self):
        self.deskewer = DocumentDeskewer()
        self.enhancer = DocumentEnhancer()
        self.segmenter = LineSegmenter()
        self.pipeline = ScanPreprocessingPipeline(output_dir="data/processed/test_run")

    def test_deskewing_angle_detection(self):
        """Prüft, ob ein künstlich verdrehtes Bild korrekt erkannt und begradigt wird."""
        # Testbild mit horizontalen Balken
        img = Image.new("L", (400, 400), 255)
        arr = np.array(img)
        # Zeichne 5 dicke horizontale Balken
        for y in [80, 140, 200, 260, 320]:
            arr[y:y+12, 50:350] = 30
        synth_img = Image.fromarray(arr)

        # Rotiere um 2.0 Grad
        rotated_img = synth_img.rotate(2.0, resample=Image.Resampling.BICUBIC, fillcolor=255)

        detected_angle = self.deskewer.detect_skew_angle(rotated_img)
        # Der erkannte Korrekturwinkel sollte nahe bei -2.0° liegen
        self.assertAlmostEqual(detected_angle, -2.0, delta=0.5)

    def test_enhancement_output_shape(self):
        """Prüft, ob die Bildverbesserung das Format und die Dimensionen beibehält."""
        img = Image.new("RGB", (300, 200), (240, 230, 210))
        enhanced = self.enhancer.enhance(img)
        self.assertEqual(enhanced.size, (300, 200))

    def test_pipeline_execution(self):
        """Prüft den kompletten Pipeline-Durchlauf auf einer vorhandenen Datei."""
        sample_path = Path("data/synthetic/historical_pretraining/demo/bonn_1948_bleistift.png")
        if sample_path.exists():
            res = self.pipeline.process_document(str(sample_path), document_id="test_doc")
            self.assertGreater(res["num_lines_extracted"], 0)
            self.assertIn("lines", res)
            self.assertTrue(Path(res["manifest_path"]).exists())


if __name__ == "__main__":
    unittest.main()
