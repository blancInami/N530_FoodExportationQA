"""Evaluate VLM-based multimodal breakdown extraction against human-annotated JSON.

Usage:
    python tools/evaluate_breakdown_vlm.py --input tests/data/1_輸日禽畜肉品.docx --expected tests/data/1_輸日禽畜肉品_解析結果.json
    python tools/evaluate_breakdown_vlm.py --input tests/data/2_歐盟蛋製品問卷.docx --expected tests/data/2_歐盟蛋製品問卷_解析結果.json
    python tools/evaluate_breakdown_vlm.py --input tests/data/3_澳洲水產品.md --expected tests/data/3_澳洲水產品_解析結果.json
"""
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

from app.services.breakdown_vlm import extract_questions_from_file_vlm


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("人工答案必須是 JSON 陣列")
    return data


async def evaluate_vlm(input_path: Path, expected_path: Path) -> dict:
    started_at = time.perf_counter()
    file_bytes = input_path.read_bytes()

    extraction_started_at = time.perf_counter()
    actual_items = await extract_questions_from_file_vlm(file_bytes, input_path.name)
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
        "extraction_mode": "vlm_multimodal",
        "expected_count": len(expected_ids),
        "actual_count": len(actual_ids),
        "true_positive_count": len(true_positives),
        "id_precision": round(precision, 4),
        "id_recall": round(recall, 4),
        "id_f1": round(f1, 4),
        "missing_ids": sorted(expected_ids - actual_ids),
        "extra_ids": sorted(actual_ids - expected_ids),
        "normalized_text_mismatch_ids": text_mismatches,
        "extraction_ms": round(extraction_ms, 1),
        "total_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="問卷原始檔案 (.docx / .pdf)")
    parser.add_argument("--expected", required=True, type=Path, help="人工答案 JSON")
    parser.add_argument("--min-recall", type=float, default=0.80, help="題號 recall 門檻")
    parser.add_argument("--report", type=Path, help="選填：寫入 JSON 報告的路徑")
    args = parser.parse_args()

    report = asyncio.run(evaluate_vlm(args.input, args.expected))
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.write_text(output + "\n", encoding="utf-8")

    return 0 if report["id_recall"] >= args.min_recall else 1


if __name__ == "__main__":
    from app.logging_config import setup_logging
    setup_logging()
    raise SystemExit(main())
