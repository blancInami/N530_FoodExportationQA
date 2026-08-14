"""Evaluate deterministic breakdown extraction against human-annotated JSON."""
import asyncio
import argparse
import json
import re
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.breakdown import (
    _strip_table_of_contents_with_metadata,
    convert_file_to_markdown,
    extract_questions,
    extract_structured_questions,
)
from app.services.breakdown_v2 import extract_questions as extract_questions_v2
from app.services.breakdown_vlm import extract_questions_from_file_vlm
from app.services.breakdown_langgraph import extract_questions_langgraph
from app.logging_config import setup_logging


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("人工答案必須是 JSON 陣列")
    return data


def evaluate(input_path: Path, expected_path: Path, mode: str = "structured") -> dict:
    started_at = time.perf_counter()
    conversion_started_at = time.perf_counter()
    markdown = convert_file_to_markdown(str(input_path), input_path.suffix)
    conversion_ms = (time.perf_counter() - conversion_started_at) * 1000
    markdown, toc_lines_removed, toc_detection_strategy = _strip_table_of_contents_with_metadata(markdown)

    extraction_started_at = time.perf_counter()
    if mode == "langgraph":
        actual_items = asyncio.run(extract_questions_langgraph(input_path.read_bytes(), input_path.name))
    elif mode == "vlm":
        actual_items = asyncio.run(extract_questions_from_file_vlm(input_path.read_bytes(), input_path.name))
    elif mode == "v2":
        actual_items = asyncio.run(extract_questions_v2(markdown))
    elif mode == "hybrid":
        actual_items = asyncio.run(extract_questions(markdown))
    else:
        actual_items = extract_structured_questions(markdown)
    extraction_ms = (time.perf_counter() - extraction_started_at) * 1000
    expected_items = _load_items(expected_path)

    actual_by_id = {str(item["question_id"]): str(item["question_text"]) for item in actual_items}
    expected_by_id = {str(item["question_id"]): str(item["question_text"]) for item in expected_items}
    actual_ids = set(actual_by_id)
    expected_ids = set(expected_by_id)
    true_positives = actual_ids & expected_ids
    precision = len(true_positives) / len(actual_ids) if actual_ids else 0.0
    recall = len(true_positives) / len(expected_ids) if expected_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    text_mismatches = [
        question_id
        for question_id in sorted(true_positives)
        if _normalize_text(actual_by_id[question_id]) != _normalize_text(expected_by_id[question_id])
    ]

    return {
        "input_file": str(input_path),
        "expected_file": str(expected_path),
        "extraction_mode": mode,
        "expected_count": len(expected_ids),
        "actual_count": len(actual_ids),
        "true_positive_count": len(true_positives),
        "id_precision": precision,
        "id_recall": recall,
        "id_f1": f1,
        "missing_ids": sorted(expected_ids - actual_ids),
        "extra_ids": sorted(actual_ids - expected_ids),
        "normalized_text_mismatch_ids": text_mismatches,
        "toc_lines_removed": toc_lines_removed,
        "toc_detection_strategy": toc_detection_strategy,
        "conversion_ms": conversion_ms,
        "structured_extraction_ms": extraction_ms,
        "total_ms": (time.perf_counter() - started_at) * 1000,
        "llm_calls": 0 if mode == "structured" else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="問卷原始檔案")
    parser.add_argument("--expected", required=True, type=Path, help="人工答案 JSON")
    parser.add_argument(
        "--mode", choices=("structured", "hybrid", "v2", "vlm", "langgraph"), default="structured",
        help="structured 僅測試表格快速路徑；hybrid 執行完整 v1 parser + LLM 管線；v2 執行 LLM-First 管線；vlm 執行視覺多模態管線；langgraph 執行擬人化循序閱讀管線",
    )
    parser.add_argument("--min-recall", type=float, default=0.95, help="題號 recall 門檻")
    parser.add_argument("--report", type=Path, help="選填：寫入 JSON 報告的路徑")
    args = parser.parse_args()

    report = evaluate(args.input, args.expected, args.mode)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.write_text(output + "\n", encoding="utf-8")

    return 0 if report["id_recall"] >= args.min_recall else 1

if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())