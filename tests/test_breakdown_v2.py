"""Unit tests for breakdown_v2 (LLM-First 問卷解析 - 方案 A 全域骨架與祖先路徑).

These tests validate:
1. JSON cleaning and validation
2. Deduplication across chunks
3. Semantic chunking with start/end positions
4. Skeleton outline candidate extraction & ancestor path binding
5. Full extraction pipeline with Mock LLM (Pass 1 Skeleton -> Pass 2 Extract -> Pass 3 Verify)
"""
import json
import re
import unittest
from unittest.mock import AsyncMock, patch

from app.services.breakdown_v2 import (
    MarkdownChunk,
    OutlineNode,
    _clean_llm_json,
    _deduplicate_items,
    _validate_extraction,
    _extract_heading_candidates,
    _bind_ancestor_paths,
    split_markdown_semantically,
    extract_questions,
)


class CleanLlmJsonTests(unittest.TestCase):
    """Tests for _clean_llm_json — code fence stripping."""

    def test_strips_json_code_fence(self):
        raw = '```json\n[{"question_id": "1", "question_text": "test"}]\n```'
        result = _clean_llm_json(raw)
        self.assertEqual(result, '[{"question_id": "1", "question_text": "test"}]')

    def test_strips_plain_code_fence(self):
        raw = '```\n[{"question_id": "1", "question_text": "test"}]\n```'
        result = _clean_llm_json(raw)
        self.assertEqual(result, '[{"question_id": "1", "question_text": "test"}]')

    def test_passes_through_clean_json(self):
        raw = '[{"question_id": "1", "question_text": "test"}]'
        result = _clean_llm_json(raw)
        self.assertEqual(result, raw)

    def test_strips_whitespace(self):
        raw = '  \n  [{"question_id": "1", "question_text": "test"}]  \n  '
        result = _clean_llm_json(raw)
        self.assertEqual(result, '[{"question_id": "1", "question_text": "test"}]')


class ValidateExtractionTests(unittest.TestCase):
    """Tests for _validate_extraction — output format validation."""

    def test_accepts_valid_array(self):
        data = [
            {"question_id": "1.1", "question_text": "What is X?"},
            {"question_id": "1.2", "question_text": "What is Y?"},
        ]
        result = _validate_extraction(data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["question_id"], "1.1")
        self.assertEqual(result[1]["question_text"], "What is Y?")

    def test_rejects_non_array(self):
        with self.assertRaises(ValueError):
            _validate_extraction({"question_id": "1", "question_text": "test"})

    def test_rejects_non_dict_item(self):
        with self.assertRaises(ValueError):
            _validate_extraction(["not a dict"])

    def test_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            _validate_extraction([{"question_id": "1"}])

    def test_filters_empty_id(self):
        data = [
            {"question_id": "1.1", "question_text": "Valid"},
            {"question_id": "", "question_text": "Empty ID"},
            {"question_id": "1.2", "question_text": "Also valid"},
        ]
        result = _validate_extraction(data)
        self.assertEqual(len(result), 2)

    def test_filters_empty_text(self):
        data = [
            {"question_id": "1.1", "question_text": "Valid"},
            {"question_id": "1.2", "question_text": ""},
        ]
        result = _validate_extraction(data)
        self.assertEqual(len(result), 1)

    def test_strips_whitespace(self):
        data = [{"question_id": "  1.1  ", "question_text": "  What is X?  "}]
        result = _validate_extraction(data)
        self.assertEqual(result[0]["question_id"], "1.1")
        self.assertEqual(result[0]["question_text"], "What is X?")

    def test_coerces_numeric_id_to_string(self):
        data = [{"question_id": 1, "question_text": "What is X?"}]
        result = _validate_extraction(data)
        self.assertEqual(result[0]["question_id"], "1")


class DeduplicateItemsTests(unittest.TestCase):
    """Tests for _deduplicate_items — duplicate question_id merging."""

    def test_no_duplicates_unchanged(self):
        items = [
            {"question_id": "1.1", "question_text": "A"},
            {"question_id": "1.2", "question_text": "B"},
        ]
        result = _deduplicate_items(items)
        self.assertEqual(len(result), 2)

    def test_merges_duplicate_ids(self):
        items = [
            {"question_id": "1.1", "question_text": "First part."},
            {"question_id": "1.1", "question_text": "Second part."},
        ]
        result = _deduplicate_items(items)
        self.assertEqual(len(result), 1)
        self.assertIn("First part.", result[0]["question_text"])
        self.assertIn("Second part.", result[0]["question_text"])

    def test_does_not_merge_identical_text(self):
        items = [
            {"question_id": "1.1", "question_text": "Same text."},
            {"question_id": "1.1", "question_text": "Same text."},
        ]
        result = _deduplicate_items(items)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question_text"], "Same text.")

    def test_preserves_insertion_order(self):
        items = [
            {"question_id": "2", "question_text": "B"},
            {"question_id": "1", "question_text": "A"},
            {"question_id": "3", "question_text": "C"},
        ]
        result = _deduplicate_items(items)
        self.assertEqual([r["question_id"] for r in result], ["2", "1", "3"])


class SkeletonAndAncestorPathTests(unittest.TestCase):
    """Tests for Pass 1 skeleton extraction & chunk ancestor path binding."""

    def test_extract_heading_candidates(self):
        md = (
            "# Chapter I. Introduction\n"
            "Some text here.\n\n"
            "## 1. Scope\n"
            "Details...\n\n"
            "**2. Target Species**\n"
            "More details..."
        )
        candidates = _extract_heading_candidates(md)
        self.assertGreaterEqual(len(candidates), 3)
        lines = [c[1] for c in candidates]
        self.assertTrue(any("Chapter I" in l for l in lines))
        self.assertTrue(any("1. Scope" in l for l in lines))
        self.assertTrue(any("2. Target Species" in l for l in lines))

    def test_bind_ancestor_paths(self):
        chunks = [
            MarkdownChunk(text="Chunk 1 content", start_pos=0, end_pos=100),
            MarkdownChunk(text="Chunk 2 content", start_pos=150, end_pos=300),
        ]
        skeleton = [
            OutlineNode(canonical_id="I", heading_title="Intro", source_position=0),
            OutlineNode(canonical_id="II", heading_title="Controls", source_position=120),
        ]
        _bind_ancestor_paths(chunks, skeleton)
        self.assertEqual(chunks[0].active_ancestor_path, "I")
        self.assertEqual(chunks[1].active_ancestor_path, "II")


class SplitMarkdownSemanticallyTests(unittest.TestCase):
    """Tests for split_markdown_semantically."""

    def test_single_small_section(self):
        md = "# Title\nSome content here."
        chunks = split_markdown_semantically(md)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Title", chunks[0].text)

    def test_multiple_sections_under_limit(self):
        md = "# Section 1\nContent 1.\n\n# Section 2\nContent 2."
        chunks = split_markdown_semantically(md)
        self.assertEqual(len(chunks), 1)

    def test_large_sections_split(self):
        section_a = "# Section A\n" + "A " * 3500
        section_b = "# Section B\n" + "B " * 3500
        md = section_a + "\n" + section_b
        chunks = split_markdown_semantically(md)
        self.assertGreaterEqual(len(chunks), 2)

    def test_no_headings_falls_back_to_paragraph_split(self):
        md = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = split_markdown_semantically(md)
        self.assertGreaterEqual(len(chunks), 1)

    def test_last_parent_heading_tracked(self):
        md = "## 2.2 Standards\nContent.\n\n### 2.2.1 Water\nMore content."
        chunks = split_markdown_semantically(md)
        self.assertEqual(len(chunks), 1)
        self.assertIn("2.2", chunks[0].last_parent_heading)

    def test_pre_heading_content_preserved(self):
        md = "Some intro text.\n\n# Section 1\nContent."
        chunks = split_markdown_semantically(md)
        full_text = " ".join(c.text for c in chunks)
        self.assertIn("Some intro text", full_text)


class ExtractQuestionsIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the full pipeline with Scheme A (Skeleton + Ancestor Path)."""

    @patch("app.services.breakdown_v2.chat_completion", new_callable=AsyncMock)
    async def test_simple_extraction_pipeline(self, mock_llm):
        """Single-chunk document skips Pass 1 skeleton and extracts directly."""
        mock_llm.return_value = json.dumps([
            {"question_id": "1", "question_text": "What is the name?"},
            {"question_id": "1.1", "question_text": "Contact details?"},
        ])

        result = await extract_questions("# Section 1\nSome question content.")

        self.assertGreaterEqual(len(result), 2)
        ids = {item["question_id"] for item in result}
        self.assertIn("1", ids)
        self.assertIn("1.1", ids)

    @patch("app.services.breakdown_v2.chat_completion", new_callable=AsyncMock)
    async def test_scheme_a_ancestor_path_injection_across_chunks(self, mock_llm):
        """Multi-chunk questionnaire invokes Pass 1 skeleton and injects Active_Ancestor_Path into Pass 2."""
        async def mock_chat(prompt, system_prompt=None, **kwargs):
            # Pass 1: Skeleton extraction
            if "Heading_Candidates" in prompt:
                return json.dumps([
                    {"canonical_id": "Chapter_I", "heading_title": "Section A", "raw_marker": "# Section A"},
                    {"canonical_id": "Chapter_II", "heading_title": "Section B", "raw_marker": "# Section B"},
                ])
            # Pass 2: Chunk A extraction
            if "Section A" in prompt:
                # LLM should receive active ancestor path
                self.assertIn("Active_Ancestor_Path", prompt)
                return json.dumps([
                    {"question_id": "Chapter_I.1", "question_text": "Question in chapter 1."},
                ])
            # Pass 2: Chunk B extraction
            if "Section B" in prompt:
                self.assertIn("Active_Ancestor_Path", prompt)
                return json.dumps([
                    {"question_id": "Chapter_II.1", "question_text": "Question in chapter 2."},
                ])
            # Pass 3: Consistency verification
            return json.dumps([
                {"question_id": "Chapter_I.1", "question_text": "Question in chapter 1."},
                {"question_id": "Chapter_II.1", "question_text": "Question in chapter 2."},
            ])

        mock_llm.side_effect = mock_chat

        section_a = "# Section A\n" + "Content A. " * 3500
        section_b = "# Section B\n" + "Content B. " * 3500
        md = section_a + "\n\n" + section_b

        result = await extract_questions(md)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["question_id"], "Chapter_I.1")
        self.assertEqual(result[1]["question_id"], "Chapter_II.1")

    @patch("app.services.breakdown_v2.chat_completion", new_callable=AsyncMock)
    async def test_json_fence_in_llm_output(self, mock_llm):
        """Verify code fences in LLM output are properly stripped."""
        mock_llm.return_value = (
            '```json\n'
            '[{"question_id": "1", "question_text": "Test question."}]\n'
            '```'
        )

        result = await extract_questions("# Title\nSome content.")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question_id"], "1")

    @patch("app.services.breakdown_v2.chat_completion", new_callable=AsyncMock)
    async def test_all_chunks_fail_raises_error(self, mock_llm):
        """Verify ValueError is raised when all chunks fail."""
        mock_llm.return_value = "NOT VALID JSON"

        with self.assertRaises(ValueError):
            await extract_questions("# Title\nSome content.")

    @patch("app.services.breakdown_v2.chat_completion", new_callable=AsyncMock)
    async def test_toc_stripped_before_extraction(self, mock_llm):
        """Verify Word-style TOC is removed before LLM extraction."""
        mock_llm.return_value = json.dumps([
            {"question_id": "1", "question_text": "Actual question."},
        ])

        md = (
            "# Table of Contents\n"
            "[Section 1](#_Toc12345)\n"
            "[Section 2](#_Toc12346)\n"
            "# Section 1\n"
            "Actual question."
        )
        result = await extract_questions(md)

        call_args = mock_llm.call_args
        prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        self.assertNotIn("_Toc12345", prompt)


if __name__ == "__main__":
    unittest.main()
