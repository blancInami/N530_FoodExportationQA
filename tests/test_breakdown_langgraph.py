"""Unit tests for breakdown_langgraph (模擬人類循序閱讀、動態章節校準與任意頁記憶回讀)."""
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.services.breakdown_langgraph import (
    QuestionnaireState,
    page_routing_decision,
    _clean_json,
    _validate_extracted_questions,
    _deduplicate_items,
    _calculate_toc_anchors,
    _get_active_anchor,
    _calibrate_toc_anchors,
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

    def test_calculate_toc_anchors(self):
        """測試目錄章節起始頁與結束頁區間計算。"""
        toc = [
            {"id": "Chapter_I", "title": "General", "page": 1},
            {"id": "Chapter_II", "title": "Standards", "page": 4},
            {"id": "Chapter_III", "title": "Inspection", "page": 10},
        ]
        anchors = _calculate_toc_anchors(toc, total_pages=15)
        self.assertEqual(len(anchors), 3)
        self.assertEqual(anchors[0]["start_page"], 1)
        self.assertEqual(anchors[0]["end_page"], 3)
        self.assertEqual(anchors[1]["start_page"], 4)
        self.assertEqual(anchors[1]["end_page"], 9)
        self.assertEqual(anchors[2]["start_page"], 10)
        self.assertEqual(anchors[2]["end_page"], 15)

    def test_get_active_anchor(self):
        """測試根據頁碼檢索當前所屬章節錨點。"""
        anchors = [
            {"id": "Chapter_I", "title": "General", "start_page": 1, "end_page": 3},
            {"id": "Chapter_II", "title": "Standards", "start_page": 4, "end_page": 9},
        ]
        a1 = _get_active_anchor(anchors, page_no=2)
        self.assertEqual(a1["id"], "Chapter_I")
        a2 = _get_active_anchor(anchors, page_no=7)
        self.assertEqual(a2["id"], "Chapter_II")

    def test_calibrate_toc_anchors_on_key_page_transition(self):
        """測試關鍵頁面研讀時即時動態校準 TOC 物理起始頁。"""
        initial_anchors = [
            {"id": "Chapter_I", "title": "General", "start_page": 1, "end_page": 3, "is_calibrated": False},
            {"id": "Chapter_II", "title": "Standards", "start_page": 4, "end_page": 10, "is_calibrated": False},
        ]
        # 模擬在物理第 6 頁實際發現了 Chapter_II
        detected_chap = {"id": "Chapter_II", "title": "Standards"}
        calibrated, is_calibrated, msg = _calibrate_toc_anchors(
            anchors=initial_anchors,
            current_page=6,
            detected_chapter=detected_chap,
            total_pages=10,
        )
        self.assertTrue(is_calibrated)
        self.assertIn("calibrated from p.4 -> real physical p.6", msg)
        # Chapter_I 的結束頁應自動閉合至 6 - 1 = 5
        self.assertEqual(calibrated[0]["end_page"], 5)
        # Chapter_II 的起始頁應校正至 6
        self.assertEqual(calibrated[1]["start_page"], 6)
        self.assertTrue(calibrated[1]["is_calibrated"])


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
    """端到端狀態機推論整合測試（涵蓋動態章節校準與物理頁面回讀）。"""

    @patch("app.services.breakdown_langgraph.expand_to_image_pages", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion_vision", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion", new_callable=AsyncMock)
    async def test_full_pipeline_with_dynamic_calibration_and_back_reading(self, mock_chat, mock_vision, mock_expand):
        # 1. 模擬展開為 4 頁影像
        mock_expand.return_value = [
            {"content_b64": "dummy_b64_p1", "mime_type": "image/jpeg"},
            {"content_b64": "dummy_b64_p2", "mime_type": "image/jpeg"},
            {"content_b64": "dummy_b64_p3", "mime_type": "image/jpeg"},
            {"content_b64": "dummy_b64_p4", "mime_type": "image/jpeg"},
        ]

        # 2. 模擬視覺推論呼叫序列：
        #    - Step 1: TOC 分析 (目錄記載 Chapter_I 始於 p.1, Chapter_II 預估始於 p.2)
        #    - Step 2: Page 1 分析 (Chapter_I 內文)
        #    - Step 3: Page 2 分析 (仍為 Chapter_I 延續)
        #    - Step 4: Page 3 分析 (在物理 Page 3 實際發現 Chapter_II，動態校準！)
        #    - Step 5: Page 4 分析 (請求回讀 Chapter_II 真實物理起始頁 Page 3)
        #    - Step 6: Back-read Inspector (調取真實 Page 3 對照還原)
        mock_vision.side_effect = [
            # TOC
            json.dumps([
                {"id": "Chapter_I", "title": "General", "page": 1},
                {"id": "Chapter_II", "title": "Standards", "page": 2},
            ]),
            # Page 1
            json.dumps({
                "detected_new_chapter": None,
                "extracted_questions": [{"question_id": "Chapter_I.1", "question_text": "Intro question."}],
                "pending_context": "",
                "active_chapter": "Chapter_I",
                "request_back_read": False,
            }),
            # Page 2
            json.dumps({
                "detected_new_chapter": None,
                "extracted_questions": [{"question_id": "Chapter_I.2", "question_text": "More general question."}],
                "pending_context": "",
                "active_chapter": "Chapter_I",
                "request_back_read": False,
            }),
            # Page 3 (實際發現 Chapter_II，動態校準！)
            json.dumps({
                "detected_new_chapter": {"id": "Chapter_II", "title": "Standards"},
                "extracted_questions": [{"question_id": "Chapter_II.1", "question_text": "Standards intro."}],
                "pending_context": "",
                "active_chapter": "Chapter_II",
                "request_back_read": False,
            }),
            # Page 4 (遭遇斷層，回讀真實物理 Page 3)
            json.dumps({
                "detected_new_chapter": None,
                "extracted_questions": [],
                "pending_context": "",
                "active_chapter": "Chapter_II",
                "request_back_read": True,
                "back_read_pages": [3],
            }),
            # Back-read inspector (對照真實 Page 3 與 Page 4)
            json.dumps({
                "detected_new_chapter": None,
                "extracted_questions": [{"question_id": "Chapter_II.2", "question_text": "Restored question from real page 3 anchor."}],
                "pending_context": "",
                "active_chapter": "Chapter_II",
                "request_back_read": False,
            }),
        ]

        # 3. 模擬最終 Reducer 審查
        mock_chat.return_value = json.dumps([
            {"question_id": "Chapter_I.1", "question_text": "Intro question."},
            {"question_id": "Chapter_I.2", "question_text": "More general question."},
            {"question_id": "Chapter_II.1", "question_text": "Standards intro."},
            {"question_id": "Chapter_II.2", "question_text": "Restored question from real page 3 anchor."},
        ])

        fake_docx_bytes = b"PK\x03\x04fake_bytes"
        result = await extract_questions_langgraph(fake_docx_bytes, "sample.docx")

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["question_id"], "Chapter_I.1")
        self.assertEqual(result[1]["question_id"], "Chapter_I.2")
        self.assertEqual(result[2]["question_id"], "Chapter_II.1")
        self.assertEqual(result[3]["question_id"], "Chapter_II.2")


if __name__ == "__main__":
    unittest.main()
