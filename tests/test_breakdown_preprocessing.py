import unittest

from app.services.breakdown_preprocessing import normalize_markdown_for_breakdown


class BreakdownPreprocessingTests(unittest.TestCase):
    def test_nfkc_normalizes_visible_text_but_preserves_link_target_and_code(self) -> None:
        markdown = (
            "ＩＩ．１．１　ＡＢＣ\n"
            "[Ｆｕｌｌ　ｗｉｄｔｈ](https://example.test/ＡＢＣ)\n"
            "```text\n"
            "ＩＩ．１．１　ＡＢＣ\n"
            "```\n"
        )

        result = normalize_markdown_for_breakdown(markdown, linearize_tables=False)

        self.assertIn("II.1.1 ABC", result)
        self.assertIn("[Full width](https://example.test/ＡＢＣ)", result)
        self.assertIn("```text\nＩＩ．１．１　ＡＢＣ\n```", result)

    def test_linearizes_unambiguous_bilingual_table_rows(self) -> None:
        markdown = "| １．１ | 設備清單 | Equipment |\n| --- | --- | --- |\n"

        result = normalize_markdown_for_breakdown(markdown)

        self.assertEqual(
            result,
            '<Row index="1"><Cell column="1" role="id">1.1</Cell>'
            '<Cell column="2" role="source">設備清單</Cell>'
            '<Cell column="3" role="target">Equipment</Cell></Row>\n',
        )

    def test_linearizes_escaped_pipes_without_dropping_cell_content(self) -> None:
        markdown = "| 1 | A\\|B | English text |\n"

        result = normalize_markdown_for_breakdown(markdown)

        self.assertIn('role="unknown">A|B</Cell>', result)
        self.assertIn('role="unknown">English text</Cell>', result)