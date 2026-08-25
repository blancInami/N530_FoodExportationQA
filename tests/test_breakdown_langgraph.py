"""Unit tests for breakdown_langgraph (模擬人類循序閱讀、動態章節校準與任意頁記憶回讀)."""
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.schemas.langgraph import QuestionnaireState
from app.services.breakdown_langgraph import (
    extract_questions_langgraph,
    page_routing_decision,
)
from app.utils.question_sanitizer import (
    canonicalize_all_question_ids,
    clean_json,
    deduplicate_items,
    detect_and_convert_weak_subsections,
    filter_false_positive_items,
    repair_nested_subquestion_hierarchy,
    repair_truncated_json_array,
    validate_extracted_questions,
)
from app.utils.toc_navigator import (
    build_page_section_map,
    build_reading_memory_summary_fast,
    calibrate_toc_anchors,
    calculate_toc_anchors,
    extract_section_prefix_from_qid,
    get_active_anchor,
    match_section_name,
    reorder_sections_by_physical_page,
    resolve_question_section,
    sanitize_depiction,
    split_section_id_and_depiction,
)


class LangGraphHelperTests(unittest.TestCase):
    """測試輔助工具函式。"""

    def test_repair_truncated_json_array(self):
        """測試 repair_truncated_json_array 在 JSON 遭中途中斷或格式不全時 100% 救回已閉合之大綱節點。"""
        # Case 1: 正常完整 JSON 陣列
        clean_raw = '[{"section_id": "A", "page": 1}, {"section_id": "B", "page": 2}]'
        res1 = repair_truncated_json_array(clean_raw)
        self.assertEqual(len(res1), 2)
        self.assertEqual(res1[0]["section_id"], "A")
        self.assertEqual(res1[1]["section_id"], "B")

        # Case 2: 於第二個節點的字串屬性中途截斷 (Unterminated string 模擬)
        truncated_raw = (
            '```json\n'
            '[\n'
            '  {"section_id": "A", "title": "General", "page": 1},\n'
            '  {"section_id": "B", "title": "Specific Controls", "page": 2},\n'
            '  {"section_id": "C", "title": "Incomplete Section Title Th'
        )
        res2 = repair_truncated_json_array(truncated_raw)
        self.assertEqual(len(res2), 2)
        self.assertEqual(res2[0]["section_id"], "A")
        self.assertEqual(res2[1]["section_id"], "B")

        # Case 3: 缺少結尾的 ']' 陣列中括號
        missing_bracket = '[{"section_id": "A", "page": 1}, {"section_id": "B", "page": 2}'
        res3 = repair_truncated_json_array(missing_bracket)
        self.assertEqual(len(res3), 2)
        self.assertEqual(res3[0]["section_id"], "A")
        self.assertEqual(res3[1]["section_id"], "B")

    def test_clean_json(self):
        raw = "```json\n{\"extracted_questions\": []}\n```"
        self.assertEqual(clean_json(raw), '{"extracted_questions": []}')

    def test_validate_extracted_questions(self):
        data = [
            {"question_id": "1.1", "question_text": "Sample question."},
            {"invalid": "item"},
        ]
        result = validate_extracted_questions(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question_id"], "1.1")

    def test_deduplicate_items(self):
        items = [
            {"question_id": "1.1", "question_text": "First page text."},
            {"question_id": "1.1", "question_text": "Second page text."},
            {"question_id": "1.2", "question_text": "Another question."},
        ]
        res = deduplicate_items(items)
        self.assertEqual(len(res), 2)
        self.assertIn("First page text.", res[0]["question_text"])
        self.assertIn("Second page text.", res[0]["question_text"])

    def test_deduplicate_items_with_html_table_merging(self):
        """測試跨頁 HTML 表格自動無縫合併。"""
        items = [
            {
                "question_id": "1.1",
                "question_text": '<p>Table header:</p><table border="1"><thead><tr><th>Col1</th></tr></thead><tbody><tr><td>Row1</td></tr></tbody></table>'
            },
            {
                "question_id": "1.1",
                "question_text": '<table border="1"><thead><tr><th>Col1</th></tr></thead><tbody><tr><td>Row2</td></tr></tbody></table>'
            }
        ]
        res = deduplicate_items(items)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["question_text"].count("<table"), 1)
        self.assertIn("Row1", res[0]["question_text"])
        self.assertIn("Row2", res[0]["question_text"])

    def test_deduplicate_items_non_consecutive_misattribution_protection(self):
        """測試第 n 頁底部表格與第 n+1 頁底部表格被中間題目隔開時，不會被錯誤合併至第 n 頁題目。"""
        items = [
            # Page n bottom: 題目 1.1 + 表格 A
            {
                "question_id": "1.1",
                "question_text": '<p>Page n table A:</p><table border="1"><tbody><tr><td>Table A row</td></tr></tbody></table>'
            },
            # Page n+1 middle: 題目 1.2, 1.3
            {
                "question_id": "1.2",
                "question_text": "Middle question 1.2"
            },
            {
                "question_id": "1.3",
                "question_text": "Middle question 1.3"
            },
            # Page n+1 bottom: 獨立表格 B（若模型重複給了 1.1）
            {
                "question_id": "1.1",
                "question_text": '<p>Page n+1 table B:</p><table border="1"><tbody><tr><td>Table B row</td></tr></tbody></table>'
            },
        ]
        res = deduplicate_items(items)
        # 題目 1.1 不會被遠端污染 Table B
        self.assertNotIn("Table B row", res[0]["question_text"])
        self.assertIn("Table A row", res[0]["question_text"])
        # Table B 作為獨立項目保留，不會被直接丟棄或覆寫
        self.assertEqual(len(res), 4)
        self.assertEqual(res[1]["question_id"], "1.2")
        self.assertEqual(res[2]["question_id"], "1.3")
        self.assertIn("Table B row", res[3]["question_text"])

    def test_orphan_table_fused_into_preceding_question(self):
        """測試孤立表格/舊題號表格自動歸屬至前一個引言題目（如 C.B.25: + <table> -> C.B.25）。"""
        items = [
            {
                "question_id": "C.B.19",
                "question_text": "Are there mechanisms in place to ensure that staff is free of any conflict of interest?<br><ul><li>[ ] Yes</li><li>[ ] No</li></ul>"
            },
            {
                "question_id": "C.B.20",
                "question_text": "Briefly describe the mechanism in place and provide reference to the relevant legal/administrative provisions"
            },
            {
                "question_id": "C.B.25",
                "question_text": "Please, indicate the supervisory arrangements that the competent authority has in place in order to determine if official controls are carried out as planned and in the manner required to ensure their effectiveness:"
            },
            {
                "question_id": "C.B.19",  # 模型重複給予了 C.B.19
                "question_text": '<table border="1"><thead><tr><th>Question</th><th>Yes</th><th>No</th></tr></thead><tbody><tr><td>Does the competent authority verify?</td><td></td><td></td></tr></tbody></table>'
            }
        ]
        res = deduplicate_items(items)
        # 應只有 3 道題目，第 4 個表格自動融合進 C.B.25 中，不再產生 C.B.19.2
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0]["question_id"], "C.B.19")
        self.assertEqual(res[1]["question_id"], "C.B.20")
        self.assertEqual(res[2]["question_id"], "C.B.25")
        self.assertIn("Does the competent authority verify?", res[2]["question_text"])
        self.assertIn('<table border="1">', res[2]["question_text"])
        self.assertNotIn("Does the competent authority verify?", res[0]["question_text"])

    def test_build_page_section_map_preserves_page_chronological_order(self):
        """測試即使傳入無序的大綱節點，build_page_section_map 依然依照實體頁碼與版面垂直位置排序。"""
        scrambled_items = [
            {"page": 3, "section_id": "C", "title": "Inspection", "position": "top", "is_major_section": True},
            {"page": 1, "section_id": "A", "title": "General", "position": "top", "is_major_section": True},
            {"page": 2, "section_id": "B", "title": "Specific", "position": "top", "is_major_section": True},
        ]
        page_map, global_tree, s_order, s_dep = build_page_section_map(scrambled_items, total_pages=3)
        self.assertEqual(s_order, ["A", "B", "C"])
        self.assertEqual(global_tree[0]["section_id"], "A")
        self.assertEqual(global_tree[1]["section_id"], "B")
        self.assertEqual(global_tree[2]["section_id"], "C")

    def test_build_page_section_map_multi_sections_per_page(self):
        """測試 build_page_section_map 建立每頁多章節/小節映射與頁面傳播。"""
        raw_items = [
            {"page": 1, "section_id": "A", "subsection_id": "A.1", "title": "General", "position": "top", "is_major_section": True, "depiction": "Part A depiction"},
            {"page": 2, "section_id": "A", "subsection_id": "A.2", "title": "Requirements", "position": "top", "is_major_section": False},
            # Page 3 has two sections: top A.3 and bottom B (new major section)
            {"page": 3, "section_id": "A", "subsection_id": "A.3", "title": "Controls", "position": "top", "is_major_section": False},
            {"page": 3, "section_id": "B", "subsection_id": "B.1", "title": "General Information", "position": "bottom", "is_major_section": True, "depiction": "Part B depiction"},
        ]
        page_map, global_tree, s_order, s_dep = build_page_section_map(raw_items, total_pages=4)

        # 驗證主章節順序與 depiction
        self.assertEqual(s_order, ["A", "B"])
        self.assertIn("depiction", s_dep["A"])
        self.assertIn("depiction", s_dep["B"])

        # 驗證 Page 3 包含 2 個章節/小節
        self.assertEqual(len(page_map[3]), 2)
        self.assertEqual(page_map[3][0]["subsection_id"], "A.3")
        self.assertEqual(page_map[3][0]["position"], "top")
        self.assertEqual(page_map[3][1]["section_id"], "B")
        self.assertEqual(page_map[3][1]["position"], "bottom")
        self.assertTrue(page_map[3][1]["is_major_section"])

        # 驗證 Page 4 自動延續上一頁之最後章節 B
        self.assertEqual(len(page_map[4]), 1)
        self.assertEqual(page_map[4][0]["section_id"], "B")

    def test_build_page_section_map_filters_out_question_like_toc_items(self):
        """測試 build_page_section_map 嚴格過濾誤入目錄之題號 (如 B.1) 與問句，避免污染 section_order。"""
        fake_outline = [
            {"page": 1, "section_id": "B.1", "title": "Please indicate the competent authority name:", "is_major_section": True},
            {"page": 1, "section_id": "B.2", "title": "Please provide organization chart?", "is_major_section": True},
            {"page": 2, "section_id": "C", "title": "Competent Authority", "is_major_section": True, "depiction": "Official control"},
        ]
        page_map, global_tree, section_order, section_depictions = build_page_section_map(fake_outline, 3)

        # 驗證 B.1 與 B.2 均被排除，section_order 只有合法章節 C
        self.assertNotIn("B.1", section_order)
        self.assertNotIn("B.2", section_order)
        self.assertIn("C", section_order)
        self.assertEqual(section_order, ["C"])

    def test_reorder_sections_by_physical_page(self):
        """測試 reorder_sections_by_physical_page 嚴格根據實體物理頁碼與位置排序章節。"""
        outline_tree = [
            {"page": 2, "section_id": "B", "position": "top"},
            {"page": 3, "section_id": "C", "position": "top"},
        ]
        # 動態在第 1 頁發現 Section A
        page_map = {
            1: [{"page": 1, "section_id": "A", "position": "top"}],
            2: [{"page": 2, "section_id": "B", "position": "top"}],
            3: [{"page": 3, "section_id": "C", "position": "top"}],
        }
        known_sections = ["B", "C", "A"] # A 原本在 tail

        sorted_secs = reorder_sections_by_physical_page(outline_tree, page_map, known_sections)
        # 驗證 A 必定因為物理出現在第 1 頁而排在第 1 位
        self.assertEqual(sorted_secs, ["A", "B", "C"])

    def test_extract_section_prefix_from_qid(self):
        """測試從題號前綴確定性提取所屬區塊。"""
        sections = ["A", "B", "C"]
        self.assertEqual(extract_section_prefix_from_qid("B.1", sections), "B")
        self.assertEqual(extract_section_prefix_from_qid("B.1.a", sections), "B")
        self.assertEqual(extract_section_prefix_from_qid("A.15", sections), "A")
        self.assertEqual(extract_section_prefix_from_qid("Part B.2", ["Part A", "Part B"]), "Part B")
        self.assertEqual(extract_section_prefix_from_qid("Chapter II.3", ["Chapter I", "Chapter II"]), "Chapter II")
        self.assertIsNone(extract_section_prefix_from_qid("1.1", sections))

    def test_resolve_question_section(self):
        """測試結合題號前綴、單題 section_id 與同頁過渡狀態的綜合區塊判定。"""
        sections = ["A", "B"]
        # Case 1: 題號前綴優先（即使處於轉折前，B.1 仍強制歸入 B）
        q_b1 = {"question_id": "B.1", "question_text": "Q text"}
        sec = resolve_question_section(q_b1, page_prev_section="A", page_new_section="A", known_sections=sections, is_after_transition=False)
        self.assertEqual(sec, "B")

        # Case 2: 無前綴題號，同頁轉折前 -> A
        q_top = {"question_id": "1.1", "question_text": "Top Q"}
        sec_top = resolve_question_section(q_top, page_prev_section="A", page_new_section="B", known_sections=sections, is_after_transition=False)
        self.assertEqual(sec_top, "A")

        # Case 3: 無前綴題號，同頁轉折後 -> B
        q_bot = {"question_id": "1.2", "question_text": "Bottom Q"}
        sec_bot = resolve_question_section(q_bot, page_prev_section="A", page_new_section="B", known_sections=sections, is_after_transition=True)
        self.assertEqual(sec_bot, "B")
    def test_repair_nested_subquestion_hierarchy(self):
        """測試 repair_nested_subquestion_hierarchy 自動修補孤立 (a), (b), (i) 之父題號繼承。"""
        items = [
            {"question_id": "1.1", "question_text": "Are slaughterhouses inspected?"},
            {"question_id": "(a)", "question_text": "frequency of inspection"},
            {"question_id": "(b)", "question_text": "qualification of inspectors"},
            {"question_id": "(i)", "question_text": "veterinary qualification"},
            {"question_id": "1.2", "question_text": "Are cold stores registered?"},
        ]
        repaired = repair_nested_subquestion_hierarchy(items)
        self.assertEqual(repaired[0]["question_id"], "1.1")
        self.assertEqual(repaired[1]["question_id"], "1.1.a")
        self.assertEqual(repaired[2]["question_id"], "1.1.b")
        self.assertEqual(repaired[3]["question_id"], "1.1.b.i")
        self.assertEqual(repaired[4]["question_id"], "1.2")

    def test_detect_and_convert_weak_subsections(self):
        """測試 detect_and_convert_weak_subsections 自動辨識純文字小節標題並提升至標題棧。"""
        items = [
            {"question_id": "2.1.3", "question_text": "Storage and transport conditions"},
            {"question_id": "2.1.3.1", "question_text": "Are temperature logs maintained during storage?"},
        ]
        questions, stack = detect_and_convert_weak_subsections(items, heading_stack=["Chapter II"])
        # 標題 2.1.3 應被轉入 stack，而不作為題目輸出
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["question_id"], "2.1.3.1")
        self.assertIn("2.1.3 Storage and transport conditions", stack)

    def test_sanitize_depiction_strips_question_text_and_duplicates(self):
        """測試 sanitize_depiction 徹底剔除誤入前言的第一題題幹、問句與重複標題。"""
        raw_dep = (
            "Please, indicate who is the contact point for RASFF notifications in your country\n\n"
            "D - Notifications of the Rapid Alert System for Food and Feed (RASFF)\n\n"
            "Notifications of the Rapid Alert System for Food and Feed (RASFF)"
        )
        questions = [
            {
                "question_id": "D.1",
                "question_text": "Please, indicate who is the contact point for RASFF notifications in your country"
            },
            {
                "question_id": "D.2",
                "question_text": "Please, provide the name of the authority responsible for handling RASFF"
            }
        ]
        clean_dep = sanitize_depiction(raw_dep, section_id="D", questions=questions)

        # 驗證第一題題幹已被徹底剔除
        self.assertNotIn("Please, indicate who is the contact point", clean_dep)
        # 驗證重複的標題字串已去重為單一乾淨標題
        self.assertEqual(clean_dep, "Notifications of the Rapid Alert System for Food and Feed (RASFF)")

    def test_match_section_name_does_not_pollute_unrelated_sections(self):
        """測試 match_section_name 在未匹配時回傳空字串，絕不盲目 fallback 污染最後一個章節。"""
        sections = ["A", "B", "C"]
        self.assertEqual(match_section_name("B", sections), "B")
        self.assertEqual(match_section_name("Chapter II", ["Chapter I", "Chapter II"]), "Chapter II")
        self.assertEqual(match_section_name("Unknown_X", sections), "")
        self.assertEqual(match_section_name("", sections), "")

    def test_split_section_id_and_depiction_generalization(self):
        """測試各種國際問卷格式 (破折號、冒號、長破折號、Annex、Part、Section) 均能完美純化且絕不誤傷單字。"""
        # Case 1: 標準破折號
        sid1, dep1 = split_section_id_and_depiction("C - Competent authority(ies)")
        self.assertEqual(sid1, "C")
        self.assertEqual(dep1, "Competent authority(ies)")

        # Case 2: 冒號分隔
        sid2, dep2 = split_section_id_and_depiction("Part I: Scope and Definitions")
        self.assertEqual(sid2, "Part I")
        self.assertEqual(dep2, "Scope and Definitions")

        # Case 3: En-dash / Em-dash 分隔
        sid3, dep3 = split_section_id_and_depiction("Chapter 2 – Animal Health Measures")
        self.assertEqual(sid3, "Chapter 2")
        self.assertEqual(dep3, "Animal Health Measures")

        # Case 4: Annex
        sid4, dep4 = split_section_id_and_depiction("Annex A — Analytical Methods & Laboratories")
        self.assertEqual(sid4, "Annex A")
        self.assertEqual(dep4, "Analytical Methods & Laboratories")

        # Case 5: 包含多行前言說明
        sid5, dep5 = split_section_id_and_depiction(
            raw_id="B",
            raw_title="General Information",
            raw_depiction="This part applies to official control bodies responsible for export certification.",
        )
        self.assertEqual(sid5, "B")
        self.assertIn("General Information", dep5)
        self.assertIn("This part applies to official control bodies", dep5)

    def test_sanitize_depiction_collision_filtering(self):
        """測試 sanitize_depiction 精準過濾撞題題幹並保留完整標題說明。"""
        raw_dep = (
            "Competent authority(ies)\n\n"
            "Please indicate who is the contact point for RASFF notifications\n\n"
            "This section applies to all official authorities."
        )
        questions = [
            {"question_id": "C.1", "question_text": "Please indicate who is the contact point for RASFF notifications in your country"}
        ]
        cleaned = sanitize_depiction(raw_dep, section_id="C", questions=questions)
        self.assertIn("Competent authority(ies)", cleaned)
        self.assertIn("This section applies to all official authorities.", cleaned)
        self.assertNotIn("Please indicate who is the contact point", cleaned)



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
    async def test_full_pipeline_with_multi_section_depiction(self, mock_chat, mock_vision, mock_expand):
        """測試多 Part 問卷的純化區塊 Key（如 'B'）與標題轉移至 depiction 提取。"""
        mock_expand.return_value = [
            {"content_b64": "dummy_p1", "mime_type": "image/jpeg"},
            {"content_b64": "dummy_p2", "mime_type": "image/jpeg"},
        ]

        mock_vision.side_effect = [
            # TOC
            json.dumps([
                {"id": "A", "title": "Bivalve Molluscs", "page": 1},
                {"id": "B", "title": "B - General Information", "page": 2},
            ]),
            # Page 1: Part A with depiction
            json.dumps({
                "detected_new_chapter": {
                    "id": "A",
                    "title": "Bivalve Molluscs",
                    "depiction": "This part applies to live bivalve molluscs produced for export."
                },
                "extracted_questions": [
                    {"question_id": "A.1", "question_text": "Are growing areas classified?"},
                    {"question_id": "A.2", "question_text": "Are harvesting controls in place?"},
                ],
                "pending_context": "",
                "active_chapter": "A",
                "request_back_read": False,
            }),
            # Page 2: Part B with title "B - General Information"
            json.dumps({
                "detected_new_chapter": {
                    "id": "B",
                    "title": "B - General Information",
                    "depiction": "Please indicate competent authority details."
                },
                "extracted_questions": [
                    {"question_id": "B.1", "question_text": "Please, indicate the name of your country:"},
                    {"question_id": "B.2", "question_text": "Please, indicate the name and address of the competent authority:"},
                ],
                "pending_context": "",
                "active_chapter": "B",
                "request_back_read": False,
            }),
        ]

        # Reducer: 分區塊審查 (純化後的 Section Key)
        mock_chat.side_effect = [
            # Section A batch
            json.dumps([{
                "section_name": "A",
                "depiction": "Bivalve Molluscs\n\nThis part applies to live bivalve molluscs produced for export.",
                "questions": [
                    {"question_id": "A.1", "question_text": "Are growing areas classified?"},
                    {"question_id": "A.2", "question_text": "Are harvesting controls in place?"},
                ]
            }]),
            # Section B batch
            json.dumps([{
                "section_name": "B",
                "depiction": "General Information\n\nPlease indicate competent authority details.",
                "questions": [
                    {"question_id": "B.1", "question_text": "Please, indicate the name of your country:"},
                    {"question_id": "B.2", "question_text": "Please, indicate the name and address of the competent authority:"},
                ]
            }]),
        ]

        fake_docx_bytes = b"PK\x03\x04fake_bytes"
        result = await extract_questions_langgraph(fake_docx_bytes, "sample_multi_part.docx")

        # 驗證區塊數量
        self.assertEqual(len(result), 2)

        # 驗證 Section A: Key 為純化的 "A"
        part_a_block = result[0]
        self.assertIn("A", part_a_block)
        part_a_detail = part_a_block["A"]
        self.assertIn("bivalve molluscs", part_a_detail["depiction"].lower())
        self.assertEqual(len(part_a_detail["questions"]), 2)
        self.assertEqual(part_a_detail["questions"][0]["question_id"], "A.1")

        # 驗證 Section B: Key 為純化的 "B" (非 "B: B - General Information")
        part_b_block = result[1]
        self.assertIn("B", part_b_block)
        part_b_detail = part_b_block["B"]
        # 原標題中的 "General Information" 正確移入 depiction
        self.assertIn("general information", part_b_detail["depiction"].lower())
        self.assertEqual(len(part_b_detail["questions"]), 2)
        self.assertEqual(part_b_detail["questions"][0]["question_id"], "B.1")
        self.assertEqual(part_b_detail["questions"][1]["question_id"], "B.2")



    @patch("app.services.breakdown_langgraph.expand_to_image_pages", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion_vision", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion", new_callable=AsyncMock)
    async def test_mid_page_section_transition_split(self, mock_chat, mock_vision, mock_expand):
        """測試同一頁內出現『前一區塊題目在上 + 新區塊題目在下』時能精準切分，不被全部歸類至新區塊。"""
        mock_expand.return_value = [
            {"content_b64": "dummy_p1", "mime_type": "image/jpeg"},
            {"content_b64": "dummy_p2", "mime_type": "image/jpeg"},
        ]

        mock_vision.side_effect = [
            # TOC
            json.dumps([
                {"id": "A", "title": "General Standards", "page": 1},
                {"id": "B", "title": "Specific Controls", "page": 1},
            ]),
            # Page 1: 同一頁頂部為 Part A 結尾題 (A.15, A.16)，中段出現 Section B 橫幅，底部為 B.1, B.2
            json.dumps({
                "detected_new_chapter": {
                    "id": "B",
                    "title": "Specific Controls",
                    "depiction": "Controls for specific products.",
                    "first_question_id": "B.1"
                },
                "extracted_questions": [
                    {"question_id": "A.15", "question_text": "Are general records kept?", "section_id": "A"},
                    {"question_id": "A.16", "question_text": "Are audit trails maintained?", "section_id": "A"},
                    {"question_id": "B.1", "question_text": "Please indicate product types:", "section_id": "B"},
                    {"question_id": "B.2", "question_text": "Please indicate temperature limits:", "section_id": "B"},
                ],
                "pending_context": "",
                "active_chapter": "B",
                "request_back_read": False,
            }),
            # Page 2: 額外題目
            json.dumps({
                "detected_new_chapter": None,
                "extracted_questions": [
                    {"question_id": "B.3", "question_text": "Please indicate packaging rules:", "section_id": "B"},
                ],
                "pending_context": "",
                "active_chapter": "B",
                "request_back_read": False,
            }),
        ]

        # Reducer: 兩個區塊各自審查
        mock_chat.side_effect = [
            # Section A batch (必須包含 A.15, A.16)
            json.dumps([{
                "section_name": "A",
                "depiction": "General Standards",
                "questions": [
                    {"question_id": "A.15", "question_text": "Are general records kept?"},
                    {"question_id": "A.16", "question_text": "Are audit trails maintained?"},
                ]
            }]),
            # Section B batch (必須包含 B.1, B.2, B.3)
            json.dumps([{
                "section_name": "B",
                "depiction": "Specific Controls\n\nControls for specific products.",
                "questions": [
                    {"question_id": "B.1", "question_text": "Please indicate product types:"},
                    {"question_id": "B.2", "question_text": "Please indicate temperature limits:"},
                    {"question_id": "B.3", "question_text": "Please indicate packaging rules:"},
                ]
            }]),
        ]

        fake_docx_bytes = b"PK\x03\x04fake_bytes"
        result = await extract_questions_langgraph(fake_docx_bytes, "sample_mid_page_split.docx")

        self.assertEqual(len(result), 2)

        # 驗證 Section A 精準保留了同頁上半部的題目
        self.assertIn("A", result[0])
        a_questions = result[0]["A"]["questions"]
        self.assertEqual(len(a_questions), 2)
        self.assertEqual(a_questions[0]["question_id"], "A.15")
        self.assertEqual(a_questions[1]["question_id"], "A.16")

        # 驗證 Section B 精準收錄同頁下半部的題目
        self.assertIn("B", result[1])
        b_questions = result[1]["B"]["questions"]
        self.assertEqual(len(b_questions), 3)
        self.assertEqual(b_questions[0]["question_id"], "B.1")
        self.assertEqual(b_questions[1]["question_id"], "B.2")


    @patch("app.services.breakdown_langgraph.expand_to_image_pages", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion_vision", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion", new_callable=AsyncMock)
    async def test_no_toc_questionnaire_with_skeleton_scan(self, mock_chat, mock_vision, mock_expand):
        """測試無目錄頁問卷透過輕量視覺骨架掃描 (Skeleton Fast-Scan) 建立同頁多章節地圖並完成解析。"""
        mock_expand.return_value = [
            {"content_b64": "p1_img", "mime_type": "image/jpeg"},
            {"content_b64": "p2_img", "mime_type": "image/jpeg"},
            {"content_b64": "p3_img", "mime_type": "image/jpeg"},
            {"content_b64": "p4_img", "mime_type": "image/jpeg"},
        ]

        mock_vision.side_effect = [
            # 1. TOC check on first 3 pages (前置 1~3 頁掃描無目錄)
            json.dumps([]),
            # 2. Page Analyzer: Page 1 (動態辨識第 1 頁為 Section A)
            json.dumps({
                "detected_new_chapter": {
                    "id": "A",
                    "title": "General Introduction",
                    "depiction": "General Introduction",
                    "first_question_id": "A.1"
                },
                "extracted_questions": [{"question_id": "A.1", "question_text": "Is CA established?", "section_id": "A"}],
                "pending_context": "",
                "active_chapter": "A",
                "request_back_read": False,
            }),
            # 4. Page Analyzer: Page 2
            json.dumps({
                "detected_new_chapter": None,
                "extracted_questions": [{"question_id": "A.2", "question_text": "Are facilities approved?", "section_id": "A"}],
                "pending_context": "",
                "active_chapter": "A",
                "request_back_read": False,
            }),
            # 5. Page Analyzer: Page 3 (同頁切分: 上半部 A.3, 下半部 B.1)
            json.dumps({
                "detected_new_chapter": {
                    "id": "B",
                    "title": "Specific Standards",
                    "depiction": "Specific Product Rules",
                    "first_question_id": "B.1"
                },
                "extracted_questions": [
                    {"question_id": "A.3", "question_text": "Are hygiene controls verified?", "section_id": "A"},
                    {"question_id": "B.1", "question_text": "Please indicate product category:", "section_id": "B"},
                ],
                "pending_context": "",
                "active_chapter": "B",
                "request_back_read": False,
            }),
            # 6. Page Analyzer: Page 4
            json.dumps({
                "detected_new_chapter": None,
                "extracted_questions": [{"question_id": "B.2", "question_text": "Please specify inspection frequency:", "section_id": "B"}],
                "pending_context": "",
                "active_chapter": "B",
                "request_back_read": False,
            }),
        ]

        # Reducer batches (Section A and Section B)
        mock_chat.side_effect = [
            # Section A batch
            json.dumps([{
                "section_name": "A",
                "depiction": "General Introduction",
                "questions": [
                    {"question_id": "A.1", "question_text": "Is CA established?"},
                    {"question_id": "A.2", "question_text": "Are facilities approved?"},
                    {"question_id": "A.3", "question_text": "Are hygiene controls verified?"},
                ]
            }]),
            # Section B batch
            json.dumps([{
                "section_name": "B",
                "depiction": "Specific Product Rules",
                "questions": [
                    {"question_id": "B.1", "question_text": "Please indicate product category:"},
                    {"question_id": "B.2", "question_text": "Please specify inspection frequency:"},
                ]
            }]),
        ]

        fake_docx_bytes = b"PK\x03\x04fake_bytes"
        result = await extract_questions_langgraph(fake_docx_bytes, "no_toc_sample.docx")

        self.assertEqual(len(result), 2)
        # 驗證 Section A
        self.assertIn("A", result[0])
        self.assertEqual(len(result[0]["A"]["questions"]), 3)
        self.assertEqual(result[0]["A"]["questions"][2]["question_id"], "A.3")

        # 驗證 Section B
        self.assertIn("B", result[1])
        self.assertEqual(len(result[1]["B"]["questions"]), 2)
        self.assertEqual(result[1]["B"]["questions"][0]["question_id"], "B.1")
        self.assertEqual(result[1]["B"]["questions"][1]["question_id"], "B.2")


    @patch("app.services.breakdown_langgraph.expand_to_image_pages", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion_vision", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion", new_callable=AsyncMock)
    async def test_response_order_strictly_follows_physical_page_appearance(self, mock_chat, mock_vision, mock_expand):
        """測試即使某章節 (如 Section B) 在初期目錄遺漏，後續在 Page 2 分析出來時，最終 response 順序仍依實體頁碼 (A -> B -> C) 排序。"""
        mock_expand.return_value = [
            {"content_b64": "p1", "mime_type": "image/jpeg"},
            {"content_b64": "p2", "mime_type": "image/jpeg"},
            {"content_b64": "p3", "mime_type": "image/jpeg"},
        ]

        mock_vision.side_effect = [
            # 1. TOC 只提到了 Section A (p.1) 與 Section C (p.3)，遺漏了 Section B (p.2)
            json.dumps([
                {"section_id": "A", "title": "General", "page": 1, "is_major_section": True},
                {"section_id": "C", "title": "Final", "page": 3, "is_major_section": True},
            ]),
            # 2. Page 1: Section A
            json.dumps({
                "detected_new_chapter": None,
                "extracted_questions": [{"question_id": "A.1", "question_text": "Please provide details for A1.", "section_id": "A"}],
                "pending_context": "",
                "active_chapter": "A",
                "request_back_read": False,
            }),
            # 3. Page 2: 偵測到新章節 Section B
            json.dumps({
                "detected_new_chapter": {"id": "B", "title": "Middle", "depiction": "Middle Section", "first_question_id": "B.1"},
                "extracted_questions": [{"question_id": "B.1", "question_text": "Please provide details for B1.", "section_id": "B"}],
                "pending_context": "",
                "active_chapter": "B",
                "request_back_read": False,
            }),
            # 4. Page 3: Section C
            json.dumps({
                "detected_new_chapter": {"id": "C", "title": "Final", "depiction": "Final Section", "first_question_id": "C.1"},
                "extracted_questions": [{"question_id": "C.1", "question_text": "Please provide details for C1.", "section_id": "C"}],
                "pending_context": "",
                "active_chapter": "C",
                "request_back_read": False,
            }),
        ]

        # Reducer: 依序審查 A, B, C
        mock_chat.side_effect = [
            json.dumps([{"section_name": "A", "depiction": "General", "questions": [{"question_id": "A.1", "question_text": "Please provide details for A1."}]}]),
            json.dumps([{"section_name": "B", "depiction": "Middle Section", "questions": [{"question_id": "B.1", "question_text": "Please provide details for B1."}]}]),
            json.dumps([{"section_name": "C", "depiction": "Final Section", "questions": [{"question_id": "C.1", "question_text": "Please provide details for C1."}]}]),
        ]

        fake_docx_bytes = b"PK\x03\x04fake_bytes"
        result = await extract_questions_langgraph(fake_docx_bytes, "ordering_test.docx")

        self.assertEqual(len(result), 3)
        # 驗證順序必須嚴格為 A -> B -> C，而不是 A -> C -> B
        self.assertIn("A", result[0])
        self.assertIn("B", result[1])
        self.assertIn("C", result[2])


if __name__ == "__main__":
    unittest.main()



    @patch("app.services.breakdown_langgraph.expand_to_image_pages", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion_vision", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion", new_callable=AsyncMock)
    async def test_sections_ordered_strictly_by_physical_page_appearance(self, mock_chat, mock_vision, mock_expand):
        """測試即使某章節是在後續分析時才動態發現，最終輸出順序仍嚴格依據文件實體頁碼先後排序。"""
        mock_expand.return_value = [
            {"content_b64": "p1_img", "mime_type": "image/jpeg"},
            {"content_b64": "p2_img", "mime_type": "image/jpeg"},
        ]

        mock_vision.side_effect = [
            # 1. TOC indexer (誤以為只有 Section B 從第 2 頁開始)
            json.dumps([
                {"page": 2, "section_id": "B", "title": "Specific Standards", "position": "top", "is_major_section": True, "depiction": "Part B"},
            ]),
            # 2. Page Analyzer: Page 1 (動態發現原本漏抓的 Section A 出現在第 1 頁)
            json.dumps({
                "detected_new_chapter": {
                    "id": "A",
                    "title": "General Standards",
                    "depiction": "Part A General",
                    "first_question_id": "A.1"
                },
                "extracted_questions": [
                    {"question_id": "A.1", "question_text": "Is legislation enacted?", "section_id": "A"},
                ],
                "pending_context": "",
                "active_chapter": "A",
                "request_back_read": False,
            }),
            # 3. Page Analyzer: Page 2 (Section B)
            json.dumps({
                "detected_new_chapter": None,
                "extracted_questions": [
                    {"question_id": "B.1", "question_text": "Please indicate product category:", "section_id": "B"},
                ],
                "pending_context": "",
                "active_chapter": "B",
                "request_back_read": False,
            }),
        ]

        # Reducer: Section A batch and Section B batch
        mock_chat.side_effect = [
            # Section A batch
            json.dumps([{
                "section_name": "A",
                "depiction": "Part A General",
                "questions": [{"question_id": "A.1", "question_text": "Is legislation enacted?"}],
            }]),
            # Section B batch
            json.dumps([{
                "section_name": "B",
                "depiction": "Part B",
                "questions": [{"question_id": "B.1", "question_text": "Please indicate product category:"}],
            }]),
        ]

        fake_docx_bytes = b"PK\x03\x04fake_bytes"
        result = await extract_questions_langgraph(fake_docx_bytes, "test_order.docx")

        self.assertEqual(len(result), 2)
        # 驗證實體出現在第 1 頁的 Section A 必定排在 result[0]
        self.assertIn("A", result[0])
        # 驗證實體出現在第 2 頁的 Section B 必定排在 result[1]
        self.assertIn("B", result[1])


    @patch("app.services.breakdown_langgraph.expand_to_image_pages", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion_vision", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion", new_callable=AsyncMock)
    async def test_complex_multi_level_competent_authority_restoration(self, mock_chat, mock_vision, mock_expand):
        """
        驗證複雜多層主管機關問卷（B, C, C.A, C.B, C.C, C.D, D, F, G）流程恢復：
        1. 確定性題號前綴歸屬，杜絕 G.xx 題目污染 C.D/D/E/F。
        2. 每個章節單一獨立輸出，無重複鍵。
        3. 輸出結構完整，HTML 表格與選項完好。
        """
        mock_expand.return_value = [
            {"content_b64": "p1_img", "mime_type": "image/jpeg"},
            {"content_b64": "p2_img", "mime_type": "image/jpeg"},
            {"content_b64": "p3_img", "mime_type": "image/jpeg"},
        ]

        mock_vision.side_effect = [
            # 1. TOC indexer
            json.dumps([
                {"page": 1, "section_id": "B", "title": "General Information", "position": "top", "is_major_section": True, "depiction": "General Info"},
                {"page": 2, "section_id": "C.A", "title": "Competent authority A", "position": "top", "is_major_section": True, "depiction": "Authority A"},
                {"page": 3, "section_id": "G", "title": "Specific Standards", "position": "top", "is_major_section": True, "depiction": "Standards"},
            ]),
            # 2. Page 1: Section B
            json.dumps({
                "detected_new_chapter": {"id": "B", "title": "General Information", "depiction": "General Info", "first_question_id": "B.1"},
                "extracted_questions": [
                    {"question_id": "B.1", "question_text": "Please indicate the name of your country:"},
                    {"question_id": "B.5", "question_text": "Please indicate product: <table><tr><td>Fish</td></tr></table>"},
                ],
                "pending_context": "",
                "active_chapter": "B",
                "request_back_read": False,
            }),
            # 3. Page 2: Section C.A and C.B
            json.dumps({
                "detected_new_chapter": {"id": "C.A", "title": "Competent authority A", "depiction": "Authority A", "first_question_id": "C.A.1"},
                "extracted_questions": [
                    {"question_id": "C.A.1", "question_text": "Full name of authority A"},
                    {"question_id": "C.B.1", "question_text": "Full name of authority B"},
                ],
                "pending_context": "",
                "active_chapter": "C.A",
                "request_back_read": False,
            }),
            # 4. Page 3: Section G
            json.dumps({
                "detected_new_chapter": {"id": "G", "title": "Specific Standards", "depiction": "Standards", "first_question_id": "G.4"},
                "extracted_questions": [
                    {"question_id": "G.4", "question_text": "Do you have legally binding standards?"},
                    {"question_id": "G.40", "question_text": "Freezing treatments table: <table><tr><td>Yes</td></tr></table>"},
                ],
                "pending_context": "",
                "active_chapter": "G",
                "request_back_read": False,
            }),
        ]

        # Reducer for B, C.A, C.B, G
        mock_chat.side_effect = [
            # B
            json.dumps([{"section_name": "B", "depiction": "General Info", "questions": [
                {"question_id": "B.1", "question_text": "Please indicate the name of your country:"},
                {"question_id": "B.5", "question_text": "Please indicate product: <table><tr><td>Fish</td></tr></table>"},
            ]}]),
            # C.A
            json.dumps([{"section_name": "C.A", "depiction": "Authority A", "questions": [
                {"question_id": "C.A.1", "question_text": "Full name of authority A"},
            ]}]),
            # C.B
            json.dumps([{"section_name": "C.B", "depiction": "Authority B", "questions": [
                {"question_id": "C.B.1", "question_text": "Full name of authority B"},
            ]}]),
            # G
            json.dumps([{"section_name": "G", "depiction": "Standards", "questions": [
                {"question_id": "G.4", "question_text": "Do you have legally binding standards?"},
                {"question_id": "G.40", "question_text": "Freezing treatments table: <table><tr><td>Yes</td></tr></table>"},
            ]}]),
        ]

        fake_docx_bytes = b"PKfake_bytes"
        result = await extract_questions_langgraph(fake_docx_bytes, "complex_test.docx")

        self.assertEqual(len(result), 4)
        sec_names = [list(block.keys())[0] for block in result]
        self.assertEqual(sec_names, ["B", "C.A", "C.B", "G"])

        # 驗證 G.40 嚴格落在 Section G，絕無污染到 C.A 或 C.B
        g_questions = result[3]["G"]["questions"]
        g_qids = [q["question_id"] for q in g_questions]
        self.assertIn("G.4", g_qids)
        self.assertIn("G.40", g_qids)

        c_b_questions = result[2]["C.B"]["questions"]
        c_b_qids = [q["question_id"] for q in c_b_questions]
        self.assertEqual(c_b_qids, ["C.B.1"])

    @patch("app.services.breakdown_langgraph.expand_to_image_pages", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion_vision", new_callable=AsyncMock)
    @patch("app.services.breakdown_langgraph.chat_completion", new_callable=AsyncMock)
    async def test_same_page_multiple_subsections_with_independent_depictions(self, mock_chat, mock_vision, mock_expand):
        """
        測試當同一頁面影像中出現多個子章節橫幅 (detected_new_chapters: [C.A, C.B]) 時：
        1. 系統一次性註冊 C.A 與 C.B，並分別保留各自獨立的前言說明 (Depiction)。
        2. 題目依據 VLM 標註之 section_id (或題號) 精準歸入所屬子章節。
        3. 最終輸出獨立的 C.A 與 C.B 區塊，前言與題目各自獨立。
        """
        mock_expand.return_value = [
            {"content_b64": "p1_img", "mime_type": "image/jpeg"},
        ]

        mock_vision.side_effect = [
            # 1. TOC indexer: 目錄僅回報大項 C
            json.dumps([
                {"page": 1, "section_id": "C", "title": "Competent Authority", "position": "top", "is_major_section": True, "depiction": "General Authority Oversight"},
            ]),
            # 2. Page 1: 同頁同時出現 C.A (頂部) 與 C.B (中段) 兩個獨立子章節橫幅
            json.dumps({
                "detected_new_chapters": [
                    {
                        "id": "C.A",
                        "title": "Central Competent Authority",
                        "depiction": "Central authority is responsible for national policy.",
                        "first_question_id": "C.A.1",
                        "position": "top"
                    },
                    {
                        "id": "C.B",
                        "title": "Regional Competent Authority",
                        "depiction": "Regional authorities execute inspections locally.",
                        "first_question_id": "C.B.1",
                        "position": "middle"
                    }
                ],
                "extracted_questions": [
                    {"question_id": "C.A.1", "question_text": "Please indicate central authority contact:", "section_id": "C.A"},
                    {"question_id": "C.A.2", "question_text": "Please provide central organizational chart:", "section_id": "C.A"},
                    {"question_id": "C.B.1", "question_text": "Please indicate regional offices:", "section_id": "C.B"},
                ],
                "pending_context": "",
                "active_chapter": "C.B",
                "request_back_read": False,
            }),
        ]

        # Reducer: 接收 C.A 與 C.B 兩個獨立區塊平行審查
        mock_chat.side_effect = [
            # C.A
            json.dumps([{"section_name": "C.A", "depiction": "Central authority is responsible for national policy.", "questions": [
                {"question_id": "C.A.1", "question_text": "Please indicate central authority contact:"},
                {"question_id": "C.A.2", "question_text": "Please provide central organizational chart:"},
            ]}]),
            # C.B
            json.dumps([{"section_name": "C.B", "depiction": "Regional authorities execute inspections locally.", "questions": [
                {"question_id": "C.B.1", "question_text": "Please indicate regional offices:"},
            ]}]),
        ]

        fake_docx_bytes = b"PK  fake_bytes"
        result = await extract_questions_langgraph(fake_docx_bytes, "multi_chapter_page_test.docx")

        # 驗證產出 2 個獨立區塊 C.A 與 C.B
        self.assertEqual(len(result), 2)
        sec_names = [list(block.keys())[0] for block in result]
        self.assertEqual(sec_names, ["C.A", "C.B"])

        # 驗證 C.A 專屬前言與題目
        ca_block = result[0]["C.A"]
        self.assertIn("Central authority is responsible for national policy", ca_block["depiction"])
        self.assertEqual(len(ca_block["questions"]), 2)
        self.assertEqual(ca_block["questions"][0]["question_id"], "C.A.1")
        self.assertEqual(ca_block["questions"][1]["question_id"], "C.A.2")

        # 驗證 C.B 專屬前言與題目
        cb_block = result[1]["C.B"]
        self.assertIn("Regional authorities execute inspections locally", cb_block["depiction"])
        self.assertEqual(len(cb_block["questions"]), 1)
        self.assertEqual(cb_block["questions"][0]["question_id"], "C.B.1")
