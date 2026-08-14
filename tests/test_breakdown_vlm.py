"""Unit tests for breakdown_vlm (視覺多模態問卷解析服務 - 階層繼承與偽題目過濾優化版)."""
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.services.breakdown_vlm import (
    VlmOutlineNode,
    _clean_vlm_json,
    _validate_vlm_extraction,
    _deduplicate_items,
    _bind_page_ancestor_paths,
    extract_questions_from_file_vlm,
)


class VlmHelperTests(unittest.TestCase):
    """測試 VLM 的 JSON 清洗、驗證與去重工具。"""

    def test_clean_vlm_json(self):
        raw = "```json\n[{\"question_id\": \"1\", \"question_text\": \"Test\"}]\n```"
        self.assertEqual(_clean_vlm_json(raw), '[{"question_id": "1", "question_text": "Test"}]')

    def test_validate_vlm_extraction(self):
        data = [
            {"question_id": "Part_A.1", "question_text": "Describe scope."},
            {"question_id": "Part_A.1.a", "question_text": "Sub item."},
            {"invalid": "item"},
        ]
        result = _validate_vlm_extraction(data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["question_id"], "Part_A.1")
        self.assertEqual(result[1]["question_id"], "Part_A.1.a")

    def test_deduplicate_items(self):
        items = [
            {"question_id": "1.1", "question_text": "Page 1 content."},
            {"question_id": "1.1", "question_text": "Page 2 continued content."},
            {"question_id": "1.2", "question_text": "Independent question."},
        ]
        result = _deduplicate_items(items)
        self.assertEqual(len(result), 2)
        self.assertIn("Page 1 content.", result[0]["question_text"])
        self.assertIn("Page 2 continued content.", result[0]["question_text"])

    def test_bind_page_ancestor_paths(self):
        """測試跨頁大綱路徑關聯與計算。"""
        skeleton = [
            VlmOutlineNode(canonical_id="Chapter_I", heading_title="Intro", page_number=1),
            VlmOutlineNode(canonical_id="Chapter_II", heading_title="Standards", page_number=3),
        ]
        paths = _bind_page_ancestor_paths(total_pages=4, skeleton_nodes=skeleton)
        self.assertEqual(paths, ["Chapter_I", "Chapter_I", "Chapter_II", "Chapter_II"])


class VlmPipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """測試 VLM 端到端多模態萃取流程（Pass 1 骨架 -> Pass 2 萃取 -> Pass 3 審查）。"""

    @patch("app.services.breakdown_vlm.expand_to_image_pages", new_callable=AsyncMock)
    @patch("app.services.breakdown_vlm.chat_completion_vision", new_callable=AsyncMock)
    @patch("app.services.breakdown_vlm.chat_completion", new_callable=AsyncMock)
    async def test_vlm_pipeline_extraction(self, mock_chat, mock_vision, mock_expand):
        # 1. Mock 轉圖產出 2 頁
        mock_expand.return_value = [
            {"content_b64": "dummy_b64_page1", "mime_type": "image/jpeg"},
            {"content_b64": "dummy_b64_page2", "mime_type": "image/jpeg"},
        ]
        # 2. Mock 視覺輸出 (Pass 1 Skeleton 大綱 + Pass 2 逐頁題目萃取)
        mock_vision.side_effect = [
            # Pass 1 Skeleton
            json.dumps([
                {"canonical_id": "Chapter_I", "heading_title": "General", "page_number": 1},
                {"canonical_id": "Chapter_II", "heading_title": "Animal Health", "page_number": 2},
            ]),
            # Pass 2 Page 1
            json.dumps([
                {"question_id": "Chapter_I.1", "question_text": "First page question."},
            ]),
            # Pass 2 Page 2 (帶有 Chapter_II 祖先路徑)
            json.dumps([
                {"question_id": "Chapter_II.1", "question_text": "Second page question."},
            ]),
        ]
        # 3. Mock 一致性審核回傳 (Pass 3)
        mock_chat.return_value = json.dumps([
            {"question_id": "Chapter_I.1", "question_text": "First page question."},
            {"question_id": "Chapter_II.1", "question_text": "Second page question."},
        ])

        fake_docx_bytes = b"PK\x03\x04fake_docx_content"
        result = await extract_questions_from_file_vlm(fake_docx_bytes, "sample.docx")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["question_id"], "Chapter_I.1")
        self.assertEqual(result[1]["question_id"], "Chapter_II.1")


if __name__ == "__main__":
    unittest.main()
