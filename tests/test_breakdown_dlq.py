import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.breakdown_dlq import write_breakdown_dead_letter


class BreakdownDeadLetterTests(unittest.TestCase):
    def test_disabled_dlq_does_not_create_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dlq.jsonl"
            settings = SimpleNamespace(
                breakdown_dlq_enabled=False,
                breakdown_dlq_path=str(path),
                breakdown_dlq_include_raw_payload=False,
            )
            with patch("app.services.breakdown_dlq.get_settings", return_value=settings):
                written = asyncio.run(write_breakdown_dead_letter(
                    stage="validation",
                    document_id="document-1",
                    unit_index=0,
                    total_units=1,
                    error=ValueError("invalid source key"),
                    candidate_keys=["C001"],
                ))

            self.assertFalse(written)
            self.assertFalse(path.exists())

    def test_enabled_dlq_writes_sanitized_payload_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "dlq.jsonl"
            settings = SimpleNamespace(
                breakdown_dlq_enabled=True,
                breakdown_dlq_path=str(path),
                breakdown_dlq_include_raw_payload=False,
            )
            with patch("app.services.breakdown_dlq.get_settings", return_value=settings):
                written = asyncio.run(write_breakdown_dead_letter(
                    stage="repair_exhausted",
                    document_id="document-1",
                    unit_index=1,
                    total_units=3,
                    error="missing source keys",
                    candidate_keys=["C001", "C002"],
                    diagnostics={"missing_keys": ["C002"]},
                    raw_payload="sensitive source text",
                ))

            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(written)
            self.assertEqual(record["stage"], "repair_exhausted")
            self.assertEqual(record["candidate_keys"], ["C001", "C002"])
            self.assertIn("raw_payload_sha256", record)
            self.assertNotIn("raw_payload", record)