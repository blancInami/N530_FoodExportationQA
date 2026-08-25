import unittest
from app.utils.prompt_loader import load_prompt, get_langgraph_prompt


class PromptLoaderTests(unittest.TestCase):
    """測試 System Prompt 外部檔案載入與快取機制。"""

    def test_load_all_langgraph_prompts(self):
        prompts = [
            "toc_indexer",
            "page_analyzer",
            "back_read_inspector",
            "memory_summary",
            "reducer_quality",
        ]
        for name in prompts:
            content = get_langgraph_prompt(name)
            self.assertIsInstance(content, str)
            self.assertTrue(len(content) > 20, f"Prompt '{name}' 內容過短或為空")

    def test_page_analyzer_prompt_contains_critical_rules(self):
        content = get_langgraph_prompt("page_analyzer")
        self.assertIn("detected_new_chapter", content)
        self.assertIn("extracted_questions", content)
        self.assertIn("pending_context", content)
        self.assertIn("視覺色彩", content)

    def test_reducer_quality_prompt_contains_critical_rules(self):
        content = get_langgraph_prompt("reducer_quality")
        self.assertIn("section_name", content)
        self.assertIn("depiction", content)
        self.assertIn("questions", content)
        self.assertIn("HTML", content)

    def test_load_non_existent_prompt_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_prompt("non_existent_cat", "non_existent_file")


if __name__ == "__main__":
    unittest.main()
