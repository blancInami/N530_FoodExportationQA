import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.breakdown import (
    NumberingState,
    SourceQuestionCandidate,
    _build_source_candidates,
    _extract_single_chunk,
    _verify_source_key_coverage,
    extract_questions,
    extract_structured_questions,
    split_markdown_semantically,
    strip_table_of_contents,
)


_QUESTIONNAIRE_MARKDOWN = """\
**1. Industrial Statistics**

**1.1 Livestock Farming Status**

**1) By livestock species (over the past 3 years)**

**2. Government Organizations**

| Category | Questions | Detailed questions and request for supporting materials |
| --- | --- | --- |
| Organization | 1 | Demonstrate nationwide control. | 1.1 | Provide central authority details. |
| 1.2 | Provide local authority details. |
| Organization | 2 | Describe budget controls. | 2.1 | Provide budget information. |
"""


class StructuredQuestionExtractionTests(unittest.TestCase):
    def test_preserves_table_hierarchy_and_excludes_chapter_container(self) -> None:
        items = extract_structured_questions(_QUESTIONNAIRE_MARKDOWN)

        self.assertEqual(
            [item["question_id"] for item in items],
            ["1.1.1", "2.1", "2.1.1.1", "2.1.1.2", "2.2", "2.2.2.1"],
        )
        self.assertNotIn("2", [item["question_id"] for item in items])

    def test_fast_path_returns_structured_items_without_llm(self) -> None:
        items = asyncio.run(extract_questions(_QUESTIONNAIRE_MARKDOWN))

        self.assertEqual(items[2]["question_id"], "2.1.1.1")

    def test_non_questionnaire_markdown_uses_no_structured_items(self) -> None:
        markdown = """\
**2. A section title**

Some prose without a question table.
"""

        self.assertEqual(extract_structured_questions(markdown), [])

    def test_fast_path_accepts_fullwidth_table_numbering_after_nfkc_normalization(self) -> None:
        markdown = _QUESTIONNAIRE_MARKDOWN.replace("| Organization | 1 |", "| Organization | １ |")

        items = asyncio.run(extract_questions(markdown))

        self.assertEqual(items[1]["question_id"], "2.1")


class TableOfContentsRemovalTests(unittest.TestCase):
    _MARKDOWN_WITH_TOC = """\
Questionnaire title

目次

[**1. Outline of regulation** 3](#_Toc15647067)
[**1.1 Standards for buildings** 10](#_Toc15647078)

**I. GENERAL PROVISIONS**

See the [official guidance](https://example.test/guidance).
"""

    def test_removes_explicit_toc_links_and_preserves_first_body_line(self) -> None:
        cleaned = strip_table_of_contents(self._MARKDOWN_WITH_TOC)

        self.assertNotIn("#_Toc15647067", cleaned)
        self.assertNotIn("#_Toc15647078", cleaned)
        self.assertIn("**I. GENERAL PROVISIONS**", cleaned)
        self.assertIn("[official guidance](https://example.test/guidance)", cleaned)

    def test_keeps_markdown_unchanged_without_a_toc_heading(self) -> None:
        markdown = "[**1. A real link**](#_TocShouldRemain)\n\nQuestion body.\n"

        self.assertEqual(strip_table_of_contents(markdown), markdown)

    def test_llm_fallback_receives_markdown_without_toc(self) -> None:
        with patch("app.services.breakdown.chat_completion", new=AsyncMock(return_value="[]")) as completion:
            asyncio.run(extract_questions(self._MARKDOWN_WITH_TOC))

        prompt = completion.await_args.kwargs["prompt"]
        self.assertNotIn("#_Toc15647067", prompt)
        self.assertIn("**I. GENERAL PROVISIONS**", prompt)


class CrossChunkNumberingStateTests(unittest.TestCase):
    def test_candidate_kind_role_alias_keeps_parent_requirement_segment(self) -> None:
        candidates = [
            SourceQuestionCandidate(
                "C001", "I.1", "Outline of regulation", "parent_requirement", 1,
            ),
        ]
        response = """[
            {"source_key": "C001", "role": "parent_requirement", "question_text": "Outline of regulation"}
        ]"""

        with patch("app.services.breakdown.chat_completion", new=AsyncMock(return_value=response)):
            items = asyncio.run(_extract_single_chunk(
                "2. Outline of regulation", 0, 1,
                numbering_state=NumberingState("I", ("1",), (999,)),
                source_candidates=candidates,
            ))

        self.assertEqual(items, [{
            "question_id": "I.1",
            "question_text": "Outline of regulation",
        }])

    def test_candidate_kind_role_alias_keeps_explicit_item_segment(self) -> None:
        candidates = [
            SourceQuestionCandidate(
                "C001", "I.3.2", "Approval procedures for establishments", "explicit_item", 1,
            ),
        ]
        response = """[
            {"source_key": "C001", "role": "explicit_item", "question_text": "Approval procedures for establishments"}
        ]"""

        with patch("app.services.breakdown.chat_completion", new=AsyncMock(return_value=response)):
            items = asyncio.run(_extract_single_chunk(
                "3.2 Approval procedures for establishments", 0, 1,
                numbering_state=NumberingState("I", ("3", "2"), (999, 999)),
                source_candidates=candidates,
            ))

        self.assertEqual(items, [{
            "question_id": "I.3.2",
            "question_text": "Approval procedures for establishments",
        }])

    def test_explicit_decimal_lines_own_distinct_question_ids(self) -> None:
        markdown = """\
# **I. General provisions**

## **4. Others**

4.1 If your country exports meat and poultry meat to other foreign countries.
4.2 When you produce meat and poultry meat products for export to Japan.
4.3 Regarding the raw material mentioned in 4.2.
"""

        candidates = _build_source_candidates(markdown, NumberingState())

        self.assertEqual(
            [candidate.question_id for candidate in candidates],
            ["I.4.1", "I.4.2", "I.4.3"],
        )

    def test_repeated_markitdown_table_list_numbers_restore_distinct_subitems(self) -> None:
        markdown = """\
| 1. **Changes introduced in the updated control plan.** |

| * 1. **Contact Details:** Please confirm the competent authority. |  |
| * 1. Please provide updated production data. |  |
| * 1. If there have been changes in the prior plan, please provide details. |  |
"""

        candidates = _build_source_candidates(markdown, NumberingState())

        self.assertEqual(
            [(candidate.question_id, candidate.kind) for candidate in candidates],
            [
                ("1", "parent_requirement"),
                ("1.1", "implicit_list_item"),
                ("1.2", "implicit_list_item"),
                ("1.3", "implicit_list_item"),
            ],
        )

    def test_repeated_markitdown_table_list_items_remain_separate_after_llm_mapping(self) -> None:
        markdown = """\
| 1. **Changes introduced in the updated control plan.** |

| * 1. **Contact Details:** Please confirm the competent authority. |  |
| * 1. Please provide updated production data. |  |
| * 1. If there have been changes in the prior plan, please provide details. |  |
"""
        candidates = _build_source_candidates(markdown, NumberingState())
        response = """[
            {"source_key": "C002", "role": "implicit_list_item", "question_text": "Contact Details: Please confirm the competent authority."},
            {"source_key": "C003", "role": "question", "question_text": "Please provide updated production data."},
            {"source_key": "C004", "role": "question", "question_text": "If there have been changes in the prior plan, please provide details."}
        ]"""

        with patch("app.services.breakdown.chat_completion", new=AsyncMock(return_value=response)):
            items = asyncio.run(_extract_single_chunk(
                markdown, 0, 1, source_candidates=candidates,
            ))

        self.assertEqual(
            [item["question_id"] for item in items],
            ["1.1", "1.2", "1.3"],
        )

    def test_japanese_word_auto_numbers_do_not_replace_english_section_hierarchy(self) -> None:
        markdown = """\
# **I. General questions**

1. **施設における公的管理**

**3. Official controls in establishments**

3.1 食肉及び食鳥肉に関する施設の情報

  3.1 Information of establishments regarding meat and poultry meat

|  |
| --- |
| 1. 貴国における食肉及び食鳥肉に関する下記の施設の総数 (1)The total number of establishments. |

* 1. 施設の認定手続

3.2 Approval procedures for establishments

|  |
| --- |
| 1. と畜場及び食鳥処理場の設置は許可制であるか。 (1)Whether or not abattoirs must obtain permit? |
|  |
| --- |
| 1. 許可権者に許可の取り消し権限があるか。 (2)Whether or not authorities may cancel permit? |
|  |
| --- |
| 1. 行政機関による施設への立ち入り権限があるか。 (3)Whether or not authorities may enter for inspection? |
|  |
| --- |
| (4)追加的な手順 (4)Additional approval procedures for establishments exporting to Japan. |
|  |
| --- |
| (5)モニタリング手順 (5)Procedures for monitoring structural conditions. |
"""

        candidates = _build_source_candidates(markdown, NumberingState())

        self.assertEqual(
            [candidate.question_id for candidate in candidates],
            [
                "I.3",
                "I.3.1",
                "I.3.1.1",
                "I.3.2",
                "I.3.2.1",
                "I.3.2.2",
                "I.3.2.3",
                "I.3.2.4",
                "I.3.2.5",
            ],
        )

    def test_haccp_inline_roman_candidates_exclude_cross_references(self) -> None:
        markdown = """\
# **II. Specific questions**

## **1. Controls in abattoirs**

### 1.3 HACCP

（2） Regarding the sanitary control, the following measures shall be taken.
| a. A document describing the following items shall be prepared. (i) Measures to prevent food sanitation hazards. (ii) The processes of (i) which require monitoring. (iii) Criteria of the control measures. (iv) Method to confirm (ii). |
| b. A document describing the improvement measures upon the confirmation of c (ii) shall be prepared. |
| d. A document describing the following records shall be prepared. (i) Items regarding the confirmation of (c) (ii) (ii) Items regarding the improvement measures of (d) (iii) Items regarding the verification of (e). |
| e. According to the documents prepared based on the provisions of (a) to (d), measures necessary for public health shall be taken. |
"""

        candidates = _build_source_candidates(markdown, NumberingState())

        self.assertEqual(
            [candidate.question_id for candidate in candidates],
            [
                "II.1.1.3.2",
                "II.1.1.3.2.a",
                "II.1.1.3.2.a.i",
                "II.1.1.3.2.a.ii",
                "II.1.1.3.2.a.iii",
                "II.1.1.3.2.a.iv",
                "II.1.1.3.2.b",
                "II.1.1.3.2.d",
                "II.1.1.3.2.d.i",
                "II.1.1.3.2.d.ii",
                "II.1.1.3.2.d.iii",
                "II.1.1.3.2.e",
            ],
        )

    def test_source_id_recovery_does_not_treat_cross_reference_as_roman_subitem(self) -> None:
        markdown = """\
# **II. Specific questions**

## **1. Controls in abattoirs**

### 1.3 HACCP

（2） Regarding the sanitary control, the following measures shall be taken.
| b. A document describing the improvement measures upon the confirmation of c (ii) shall be prepared. |
"""

        from app.services.breakdown import _source_question_id

        question_id = _source_question_id(
            markdown,
            "A document describing the improvement measures upon the confirmation of c (ii) shall be prepared.",
            NumberingState(),
        )

        self.assertEqual(question_id, "II.1.1.3.2.b")

    def test_separate_questions_with_same_id_are_not_merged(self) -> None:
        response = """[
            {"question_id": "II.1.1.3.2", "question_text": "Regarding the sanitary control."},
            {"question_id": "II.1.1.3.2", "question_text": "Other than those specified in (1), additional measures shall be taken."}
        ]"""

        with patch("app.services.breakdown.chat_completion", new=AsyncMock(return_value=response)):
            items = asyncio.run(_extract_single_chunk(
                "(2) First requirement\n(2) Second requirement", 0, 1,
                numbering_state=NumberingState("II", ("1", "1", "3"), (1, 2, 3), "2"),
            ))

        self.assertEqual(
            items,
            [
                {"question_id": "II.1.1.3.2", "question_text": "Regarding the sanitary control."},
                {"question_id": "II.1.1.3.2", "question_text": "Other than those specified in (1), additional measures shall be taken."},
            ],
        )

    def test_llm_continuation_merges_into_its_parser_owned_parent_id(self) -> None:
        candidates = [
            SourceQuestionCandidate(
                "C001", "II.2.2.2.6", "Lavatory", "parent_heading", 1,
            ),
            SourceQuestionCandidate(
                "C002", "II.2.2.2.6",
                "Lavatories shall be kept clean and disinfected at regular intervals.",
                "parent_requirement", 2,
            ),
        ]
        response = """[
            {"source_key": "C001", "role": "question", "question_text": "Lavatory"},
            {"source_key": "C002", "role": "continuation", "question_text": "Lavatories shall be kept clean and disinfected at regular intervals."}
        ]"""

        with patch("app.services.breakdown.chat_completion", new=AsyncMock(return_value=response)):
            items = asyncio.run(_extract_single_chunk(
                "(6) Lavatory\nLavatories shall be kept clean and disinfected at regular intervals.",
                0, 1, numbering_state=NumberingState("II", ("2", "2", "2"), (1, 2, 3), "6"),
                source_candidates=candidates,
            ))

        self.assertEqual(items, [{
            "question_id": "II.2.2.2.6",
            "question_text": "Lavatory Lavatories shall be kept clean and disinfected at regular intervals.",
        }])

    def test_inline_numeric_enumerations_do_not_replace_parent_item(self) -> None:
        markdown = """\
# **II. Specific questions**

## **2. Controls in poultry processing plants**

### 2.2 Sanitation

(8) Fabrication
| f. (1)Both the inside and the outside of any eviscerated carcass shall be thoroughly washed. (2)Any viscera shall be divided into edible and inedible parts. |
| g. (1)Abdominal incision shall be done. (2)Any organs shall be withdrawn. |

(9) Employee
| a. (1) Requiring any employee to control their own health condition. (2)Requiring any employee not to go to the lavatory with apron. |
| b. Requiring any employee to wear clean working clothes. |
"""

        candidates = _build_source_candidates(markdown, NumberingState())

        self.assertEqual(
            [candidate.question_id for candidate in candidates],
            [
                "II.2.2.2.8",
                "II.2.2.2.8.f",
                "II.2.2.2.8.g",
                "II.2.2.2.9",
                "II.2.2.2.9.a",
                "II.2.2.2.9.b",
            ],
        )

    def test_opt_in_verification_only_reviews_anomalous_source_key_coverage(self) -> None:
        candidates = [
            SourceQuestionCandidate("C001", "II.2.2.3.2.a"),
            SourceQuestionCandidate("C002", "II.2.2.3.2.a.i"),
        ]
        verification = (
            '{"confirmed_keys": ["C001"], "missing_keys": ["C002"], '
            '"duplicate_keys": []}'
        )

        with (
            patch(
                "app.services.breakdown.get_settings",
                return_value=SimpleNamespace(breakdown_llm_verify=True),
            ),
            patch(
                "app.services.breakdown.chat_completion",
                new=AsyncMock(return_value=verification),
            ) as completion,
        ):
            result = asyncio.run(_verify_source_key_coverage(candidates, ["C001"], 0, 1))

        self.assertEqual(result.missing_keys, ("C002",))
        self.assertEqual(result.duplicate_keys, ())
        self.assertEqual(result.verification, {
            "confirmed_keys": ["C001"],
            "missing_keys": ["C002"],
            "duplicate_keys": [],
        })
        self.assertEqual(completion.await_count, 1)
        self.assertIn("C002", completion.await_args.kwargs["prompt"])

    def test_source_id_accepts_a_letter_marker_without_whitespace_before_japanese(self) -> None:
        markdown = """\
# **II. Specific questions**

## **1．Controls in slaughterhouses**

### 1-4 Controls implemented in slaughterhouses

(3) Frequency and nature of official controls
| a.地域又は国レベルによる優先順位付け  a. Prioritization system for inspections and/or audits by the Regional and/or Central level. |
"""

        responses = [
            "[]",
            "[]",
            '[{"question_id": "II.2.2.4.2.4.6.a", "question_text": "Prioritization system for inspections and/or audits by the Regional and/or Central level."}]',
        ]
        with patch("app.services.breakdown.chat_completion", new=AsyncMock(side_effect=responses)):
            items = asyncio.run(extract_questions(markdown))

        self.assertEqual(items[-1]["question_id"], "II.1.1.4.3.a")

    def test_heading_scope_owns_the_id_when_llm_returns_an_accumulated_path(self) -> None:
        markdown = """\
# **II. Specific questions**

## **2．Controls in poultry processing plants**

### 2.4 Controls implemented in poultry processing plants

(1) Routine controls
| e. | Procedures for follow-up of rectification of non-compliance. |
"""

        responses = [
            "[]",
            "[]",
            '[{"question_id": "II.2.2.4.2.4.6.e", "question_text": "Procedures for follow-up of rectification of non-compliance."}]',
        ]
        with patch("app.services.breakdown.chat_completion", new=AsyncMock(side_effect=responses)):
            items = asyncio.run(extract_questions(markdown))

        self.assertEqual(items[-1]["question_id"], "II.2.2.4.1.e")

    def test_numbered_markdown_heading_replaces_a_stale_child_hierarchy(self) -> None:
        markdown = """\
# **II. Specific questions**

## **1．Controls in slaughterhouses**

* 1. Buildings
  2. Hygiene
**(1) General provision**
| d. | Previous detail |

### 1-4 Controls implemented in slaughterhouses

(1) Routine controls
| f. | Procedures to be followed in the event of detection of food-borne pathogens. |
"""

        responses = [
            "[]",
            "[]",
            '[{"question_id": "II.1.2.1.1.4.1.f", "question_text": "Procedures to be followed in the event of detection of food-borne pathogens."}]',
        ]
        with patch("app.services.breakdown.chat_completion", new=AsyncMock(side_effect=responses)) as completion:
            items = asyncio.run(extract_questions(markdown))

        self.assertEqual(items[-1]["question_id"], "II.1.1.4.1.f")
        last_prompt = completion.await_args_list[-1].kwargs["prompt"]
        self.assertIn("父題號為 II.1.1.4", last_prompt)

    def test_preserves_roman_chapter_for_a_section_first_seen_in_the_same_chunk(self) -> None:
        markdown = """\
# **I. General provisions**

1. Japanese question text
2. Outline of regulation for Abattoir and Poultry Slaughtering Plant

| 1. A list of principal legislations. |
| 1. Please provide copies of the legislations above in English. |

# **II. Specific questions**

1. Another chapter item
"""

        responses = [
            '[{"question_id": "1", "question_text": "Outline of regulation for Abattoir and Poultry Slaughtering Plant"}, '
            '{"question_id": "1.1", "question_text": "A list of principal legislations."}, '
            '{"question_id": "1.2", "question_text": "Please provide copies of the legislations above in English."}]',
            "[]",
        ]
        with patch("app.services.breakdown.chat_completion", new=AsyncMock(side_effect=responses)) as completion:
            items = asyncio.run(extract_questions(markdown))

        self.assertEqual(
            [item["question_id"] for item in items],
            ["I.1", "I.1.1", "I.1.2"],
        )
        first_prompt = completion.await_args_list[0].kwargs["prompt"]
        self.assertIn("<Numbering_Context>", first_prompt)
        self.assertIn("父題號為 I", first_prompt)

    def test_preserves_non_heading_hierarchy_across_an_oversized_section(self) -> None:
        markdown = """\
# **II. Specific provisions**

## **1．Controls in slaughterhouses**

* 1. General matters
* 2. Standards for hygiene control
**(1) General provision**
| c. | Previous item |
""" + ("x" * 6_001) + "\n| d. | Necessary illuminance |\n"

        chunks = split_markdown_semantically(markdown)
        d_chunk = next(chunk for chunk in chunks if "Necessary illuminance" in chunk.text)
        self.assertEqual(d_chunk.start_numbering_state.parent_prefix(), "II.1.2.1")

        responses = ["[]"] * (len(chunks) - 1) + [
            '[{"question_id": "1.d", "question_text": "Necessary illuminance"}]',
        ]
        with patch("app.services.breakdown.chat_completion", new=AsyncMock(side_effect=responses)) as completion:
            items = asyncio.run(extract_questions(markdown))

        self.assertEqual(items[-1]["question_id"], "II.1.2.1.d")
        prompt = completion.await_args_list[-1].kwargs["prompt"]
        self.assertIn("<Numbering_Context>", prompt)
        self.assertIn("II.1.2.1", prompt)

    def test_source_structure_owns_fullwidth_haccp_parent_and_inline_roman_ids(self) -> None:
        markdown = """\
# **II. Specific questions**

## **2．Controls in poultry processing plants**

### 2.3 HACCP

（2） Regarding the sanitary control, the following measures shall be taken.
| a. 次に掲げる事項を記載した文書を作成すること。 a. A document describing the following items shall be prepared. (i) 危害防止措置 (i) Measures to prevent food sanitation hazards. (ii) 管理基準 (ii) Criteria of the control measures. |
| e. 公衆衛生上必要な措置を講ずること。 g. According to the documents, measures necessary for public health shall be taken. |
（3） Provide control to ensure that the measures shall be taken properly.
"""
        responses = [
            "[]",
            "[]",
            """[
                {"source_key": "C001", "question_text": "Regarding the sanitary control, the following measures shall be taken."},
                {"source_key": "C002", "question_text": "A document describing the following items shall be prepared."},
                {"source_key": "C003", "question_text": "Measures to prevent food sanitation hazards."},
                {"source_key": "C004", "question_text": "Criteria of the control measures."},
                {"source_key": "C005", "question_text": "According to the documents, measures necessary for public health shall be taken."},
                {"source_key": "C006", "question_text": "Provide control to ensure that the measures shall be taken properly."}
            ]""",
        ]

        with patch("app.services.breakdown.chat_completion", new=AsyncMock(side_effect=responses)) as completion:
            items = asyncio.run(extract_questions(markdown))

        self.assertEqual(
            [item["question_id"] for item in items],
            [
                "II.2.2.3.2",
                "II.2.2.3.2.a",
                "II.2.2.3.2.a.i",
                "II.2.2.3.2.a.ii",
                "II.2.2.3.2.e",
                "II.2.2.3.3",
            ],
        )
        prompt = completion.await_args_list[-1].kwargs["prompt"]
        self.assertIn(
            '<Candidate source_key="C003" question_id="II.2.2.3.2.a.i" '
            'kind="inline_subitem"',
            prompt,
        )


class QaBreakdownRouterTests(unittest.TestCase):
    def test_breakdown_section_detail_questions_and_question_alias(self) -> None:
        from app.schemas.breakdown import BreakdownSectionDetail, BreakdownQuestion

        detail = BreakdownSectionDetail(
            depiction="Sample depiction",
            question=[
                BreakdownQuestion(question_id="1.1", question_text="Question text")
            ]
        )
        self.assertEqual(len(detail.question), 1)
        self.assertEqual(len(detail.questions), 1)
        self.assertEqual(detail.question[0].question_id, "1.1")

        # Test initializing with questions keyword
        detail2 = BreakdownSectionDetail(
            depiction="Sample depiction 2",
            questions=[
                BreakdownQuestion(question_id="1.2", question_text="Question text 2")
            ]
        )
        self.assertEqual(len(detail2.question), 1)
        self.assertEqual(len(detail2.questions), 1)
        self.assertEqual(detail2.question[0].question_id, "1.2")
