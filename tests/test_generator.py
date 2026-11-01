"""
Automatisierte Tests für den DEK-Generator.
"""

import unittest
from pathlib import Path
from src.dek_rules.parser import DekParser
from src.dek_rules.symbols import CONSONANTS, KUERZEL, VOWEL_CONNECTORS
from src.generator.stroke_model import StrokeAssembler
from src.generator.renderer import DekRenderer
from src.generator.dataset import DekDatasetGenerator
from src.generator.page_generator import DekPageGenerator
from src.generator.historical.corpus_loader import HistoricalCorpusLoader
from src.generator.historical.post_war_augmenter import PostWarAugmenter


class TestDekGenerator(unittest.TestCase):

    def setUp(self):
        self.parser = DekParser()
        self.assembler = StrokeAssembler()
        self.renderer = DekRenderer()
        self.generator = DekDatasetGenerator(output_dir="data/synthetic/test_run")
        self.page_generator = DekPageGenerator(output_dir="data/synthetic/test_run/pages")

    def test_parser_kuerzel(self):
        """Prüft, ob Standardkürzel korrekt als solche erkannt werden."""
        token = self.parser.parse_word("und")
        self.assertTrue(token.is_kuerzel)
        self.assertEqual(len(token.glyphs), 1)
        self.assertEqual(token.glyphs[0].name, "kuerzel_und")

        token_nicht = self.parser.parse_word("nicht")
        self.assertTrue(token_nicht.is_kuerzel)
        self.assertEqual(token_nicht.glyphs[0].name, "kuerzel_nicht")

    def test_parser_word_decomposition(self):
        """Prüft die Konsonant-Vokal-Zerlegung regulärer Wörter."""
        token = self.parser.parse_word("leben")
        self.assertFalse(token.is_kuerzel)
        glyph_names = [g.name for g in token.glyphs]
        self.assertIn("l", glyph_names)
        self.assertIn("b", glyph_names)
        self.assertIn("n", glyph_names)

    def test_stroke_assembly_and_rendering(self):
        """Prüft, ob Striche fehlerfrei assembliert und gerendert werden."""
        tokens = self.parser.parse_text("der bund und die welt")
        self.assertGreaterEqual(len(tokens), 4)

        img = self.generator.generate_single("der bund und die welt", apply_augmentation=False)
        self.assertIsNotNone(img)
        self.assertGreater(img.width, 100)
        self.assertGreater(img.height, 50)

    def test_page_generation(self):
        """Prüft die Generierung ganzer Stenoseiten inklusive Zeilen-Metadaten."""
        sample_text = "das leben ist gut und der mann geht nach hause nicht mit worten sondern mit taten"
        page_img, meta = self.page_generator.generate_page(
            text=sample_text,
            has_margin_line=True,
            apply_page_skew=False
        )
        self.assertIsNotNone(page_img)
        self.assertEqual(page_img.width, 1400)
        self.assertEqual(page_img.height, 1980)
        self.assertGreater(meta["num_lines"], 0)
        self.assertIn("lines", meta)
        first_line = meta["lines"][0]
        self.assertIn("bbox", first_line)
        self.assertIn("text", first_line)

    def test_historical_components(self):
        """Prüft den historischen Korpus-Manager und Alterungseffekte."""
        corpus = HistoricalCorpusLoader()
        para = corpus.generate_paragraph(min_sentences=4, max_sentences=6)
        self.assertGreater(len(para), 20)

        # Prüfe Post-War Augmenter
        aug = PostWarAugmenter(paper_tone="aged_yellow", tool_type="pencil")
        color, thick = aug.get_tool_color_and_thickness()
        self.assertIsInstance(color, tuple)
        self.assertGreater(thick, 0.5)

        # Schneller Test auf synthetischem Mini-Bild
        from PIL import Image
        test_img = Image.new("RGB", (200, 200), (255, 255, 255))
        aged = aug.apply_historical_paper_and_aging(test_img, add_foxing=True, add_fold_crease=True)
        self.assertEqual(aged.size, (200, 200))

    def test_gabelsberger_and_stolze_schrey(self):
        """Prüft Parser und Seiten-Generierung für Gabelsberger und Stolze-Schrey."""
        for sys_name in ("gabelsberger", "stolze_schrey"):
            parser = DekParser(system=sys_name)
            token_und = parser.parse_word("und")
            self.assertTrue(token_und.is_kuerzel)
            self.assertEqual(token_und.system, sys_name)

            token_word = parser.parse_word("freiheit")
            self.assertGreater(len(token_word.glyphs), 0)

            page_gen = DekPageGenerator(output_dir="data/synthetic/test_run/pages", system=sys_name)
            img, meta = page_gen.generate_page(
                text="die freiheit und der frieden sind für das volk von großer bedeutung",
                apply_page_skew=False,
            )
            self.assertIsNotNone(img)
            self.assertGreater(meta["num_lines"], 0)


if __name__ == "__main__":
    unittest.main()
