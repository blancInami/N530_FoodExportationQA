"""
Unit tests for app.utils.html_formatter.
"""
import unittest

from app.utils.html_formatter import (
    convert_markdown_table_to_html,
    convert_markdown_checklist_to_html,
    sanitize_and_balance_html,
    merge_html_composite_texts,
    format_composite_question_text,
    strip_question_id_prefix,
    strip_fill_in_blanks,
)


class HtmlFormatterTests(unittest.TestCase):
    def test_convert_markdown_table_to_html(self):
        text = """Please review the following table:
| Establishment | Capacity | Status |
| --- | :---: | ---: |
| Plant A | 100 | Approved |
| Plant B | 200 | Pending |
"""
        html_out = convert_markdown_table_to_html(text)
        self.assertIn('<table border="1">', html_out)
        self.assertIn("<thead>", html_out)
        self.assertIn("<th>Establishment</th>", html_out)
        self.assertIn("<tbody>", html_out)
        self.assertIn("<td>Plant A</td>", html_out)
        self.assertIn("<td>Approved</td>", html_out)
        self.assertTrue(html_out.startswith("Please review the following table:"))

    def test_convert_markdown_checklist_to_html(self):
        text = """Please check the categories covered:
- [ ] Cattle
- [x] Swine
- [ ] Poultry
"""
        html_out = convert_markdown_checklist_to_html(text)
        self.assertIn("<ul>", html_out)
        self.assertIn("<li>[ ] Cattle</li>", html_out)
        self.assertIn("<li>[x] Swine</li>", html_out)
        self.assertIn("<li>[ ] Poultry</li>", html_out)

    def test_sanitize_and_balance_html(self):
        unclosed = "<p>Question text <table><thead><tr><th>Header</th></tr><tbody><tr><td>Cell"
        balanced = sanitize_and_balance_html(unclosed)
        self.assertIn("</td>", balanced)
        self.assertIn("</tr>", balanced)
        self.assertIn("</tbody>", balanced)
        self.assertIn("</table>", balanced)
        self.assertIn("</p>", balanced)

    def test_merge_html_composite_texts_tables(self):
        p1 = """<p>Facility details:</p>
<table border="1">
  <thead>
    <tr><th>ID</th><th>Name</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>Abattoir A</td></tr>
  </tbody>
</table>"""

        p2 = """<table border="1">
  <thead>
    <tr><th>ID</th><th>Name</th></tr>
  </thead>
  <tbody>
    <tr><td>2</td><td>Abattoir B</td></tr>
    <tr><td>3</td><td>Abattoir C</td></tr>
  </tbody>
</table>"""

        merged = merge_html_composite_texts(p1, p2)
        self.assertEqual(merged.count("<table"), 1)
        self.assertEqual(merged.count("</table>"), 1)
        self.assertIn("Abattoir A", merged)
        self.assertIn("Abattoir B", merged)
        self.assertIn("Abattoir C", merged)

    def test_merge_html_composite_texts_lists(self):
        l1 = """<p>Select scope:</p>
<ul>
  <li>[ ] Option 1</li>
  <li>[ ] Option 2</li>
</ul>"""
        l2 = """<ul>
  <li>[ ] Option 3</li>
  <li>[ ] Option 4</li>
</ul>"""
        merged = merge_html_composite_texts(l1, l2)
        self.assertEqual(merged.count("<ul"), 1)
        self.assertEqual(merged.count("</ul>"), 1)
        self.assertIn("Option 1", merged)
        self.assertIn("Option 4", merged)

    def test_format_composite_question_text_fallback(self):
        raw = """Please provide details:
| Animal | Limit |
| --- | --- |
| Pig | 0.05 |
"""
    def test_strip_question_id_prefix(self):
        self.assertEqual(
            strip_question_id_prefix("1.1", "1.1 Please provide registration details."),
            "Please provide registration details."
        )
        self.assertEqual(
            strip_question_id_prefix("1.1", "1.1. Please provide registration details."),
            "Please provide registration details."
        )
        self.assertEqual(
            strip_question_id_prefix("II.1.a", "II.1.a Outline of regulation"),
            "Outline of regulation"
        )
        self.assertEqual(
            strip_question_id_prefix("Part A.1", "Part A.1 - Description of facility"),
            "Description of facility"
        )
        self.assertEqual(
            strip_question_id_prefix("1.1", "<p>1.1 Please provide data.</p>"),
            "<p>Please provide data.</p>"
        )
        self.assertEqual(
            strip_question_id_prefix("1.1", "<p><strong>1.1.</strong> Please provide data.</p>"),
            "<p>Please provide data.</p>"
        )
        self.assertEqual(
            strip_question_id_prefix("1.1.2", "(2) What is the inspection procedure?"),
            "What is the inspection procedure?"
        )

    def test_strip_fill_in_blanks(self):
        # 1. 移除整段式填答底線或空白方框
        t1 = "<p>What is your official system?</p><p>[________________________________________________________________________________]</p>"
        self.assertEqual(
            strip_fill_in_blanks(t1),
            "<p>What is your official system?</p>"
        )

        t2 = "<p>Please provide details:</p><p>[____________________]</p>"
        self.assertEqual(
            strip_fill_in_blanks(t2),
            "<p>Please provide details:</p>"
        )

        # 2. 移除行內長底線
        t3 = "Please specify the volume: __________________________ kg"
        self.assertEqual(
            strip_fill_in_blanks(t3),
            "Please specify the volume:  kg"
        )

        # 3. 確保勾選框 [ ] / [x] 及其文字不會被誤刪
        t4 = "<p>Select:</p><ul><li>[ ] Option A</li><li>[x] Option B</li></ul>"
        self.assertEqual(
            strip_fill_in_blanks(t4),
            "<p>Select:</p><ul><li>[ ] Option A</li><li>[x] Option B</li></ul>"
        )

        # 4. format_composite_question_text 綜合驗證
        t5 = "1.1 What is the export standard?<p>[__________________________________________________]</p>"
        res = format_composite_question_text(t5, question_id="1.1")
        self.assertEqual(res, "What is the export standard?")


if __name__ == "__main__":
    unittest.main()
