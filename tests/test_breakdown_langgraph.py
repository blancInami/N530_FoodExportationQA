"""Unit tests for breakdown_langgraph (模擬人類循序閱讀與主動回讀問卷解析服務)."""
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.services.breakdown_langgraph import (
    QuestionnaireState,
    page_routing_decision,
    _clean_json,
    _validate_extracted_questions,
    _deduplicate_items,
    extract_questions_langgraph,
)


class LangGraphHelperTests(unittest.TestCase):
    """測試輔助工具函式。"""

    def test_clean_json(self):
        raw = "```json\n{\"extracted_questions\": []}\n```"
        self.assertEqual(_clean_json(raw), '{"extracted_questions": []}')

    def test_validate_extracted_questions(self):
        data = [
            {"question_id": "1.1", "question_text": "Sample question."},
            {"invalid": "item"},
        ]
        result = _validate_extracted_questions(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question_id"], "1.1")

    def test_deduplicate_items(self):
        items = [
            {"question_id": "1.1", "question_text": "First page text."},
            {"question_id": "1.1", "question_text": "Second page text."},
            {"question_id": "1.2", "question_text": "Another question."},
        ]
        res = _deduplicate_items(items)
        self.assertEqual(len(res), 2)
        self.assertIn("First page text.", res[0]["question_text"])
        self.assertIn("Second page text.", res[0]["question_text"])


class LangGraphRoutingTests(unittest.TestCase):
    """測試條件路由決策邏輯 (page_routing_decision)。"""

    def test_routing_advances_page_when_not_last(self):
        state: QuestionnaireState = {
            "current_index": 0,
            "total_pages": 3,
            "is_back_reading": False,
            "back_read_count_for_page": 0,
        }
        self.assertEqual(page_routing_decision(state), "advance")

    def test_routing_triggers_back_read(self):
        state: QuestionnaireState = {
            "current_index": 1,
            "total_pages": 3,
            "is_back_reading": True,
            "back_read_count_for_page": 0,
        }
        self.assertEqual(page_routing_decision(state), "back_read")

    def test_routing_prevents_infinite_back_read(self):
        state: QuestionnaireState = {
            "current_index": 1,
            "total_pages": 3,
            "is_back_reading": True,
            "back_read_count_for_page": 1,  # 已回讀過 1 次
        }
        # 超過上限應推進至下一頁
        self.assertEqual(page_routing_decision(state), "advance")

    def test_routing_finalizes_on_last_page(self):
        state: QuestionnaireState = {
            "current_index": 2,
            "total_pages": 3,
            "is_back_reading": False,
            "back_read_count_for_page": 0,
        }
        self.assertEqual(page_routing_decision(state), "finalize")


class LangGraphPipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """端到端狀態機推論整合測試。"""

    @patch("app.services.breakdown_langgraph.expand_to_image_pages", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion_vision", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion", new_callable=AsyncMock)
    async def test_full_pipeline_with_back_reading(self, mock_chat, mock_vision, mock_expand):
        # 1. 模擬展開為 2 頁影像
        mock_expand.return_value = [
            {"content_b64": "dummy_b64_p1", "mime_type": "image/jpeg"},
            {"content_b64": "dummy_b64_p2", "mime_type": "image/jpeg"},
        ]

        # 2. 模擬視覺推論呼叫序列：
        #    - Step 1: TOC 分析
        #    - Step 2: Page 1 分析 (提取 1.1，遺留 pending_context)
        #    - Step 3: Page 2 分析 (請求回讀 request_back_read=True)
        #    - Step 4: Back-Read 對照 (兩頁對照產出最終修正)
        mock_vision.side_effect = [
            # TOC
            json.dumps([{"id": "Chapter_I", "title": "General Requirements", "page": 1}]),
            # Page 1
            json.dumps({
                "extracted_questions": [{"question_id": "Chapter_I.1", "question_text": "Intro question."}],
                "pending_context": "Unfinished sentence at bottom...",
                "active_chapter": "Chapter_I",
                "request_back_read": False,
            }),
            # Page 2 (要求回讀)
            json.dumps({
                "extracted_questions": [],
                "pending_context": "",
                "active_chapter": "Chapter_I",
                "request_back_read": True,
                "back_read_pages": [1],
            }),
            # Back-read inspector (兩頁對照)
            json.dumps({
                "extracted_questions": [{"question_id": "Chapter_I.2", "question_text": "Stitched question."}],
                "pending_context": "",
                "active_chapter": "Chapter_I",
                "request_back_read": False,
            }),
        ]

        # 3. 模擬最終 Reducer 審查
        mock_chat.return_value = json.dumps([
            {"question_id": "Chapter_I.1", "question_text": "Intro question."},
            {"question_id": "Chapter_I.2", "question_text": "Stitched question."},
        ])

        fake_docx_bytes = b"PK\x03\x04fake_bytes"
        result = await extract_questions_langgraph(fake_docx_bytes, "sample.docx")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["question_id"], "Chapter_I.1")
        self.assertEqual(result[1]["question_id"], "Chapter_I.2")


if __name__ == "__main__":
    unittest.main()
