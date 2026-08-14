"""Evaluate LLM-First breakdown v2 extraction against human-annotated JSON.

Usage:
    python tools/evaluate_breakdown_v2.py --input tests/data/1_輸日禽畜肉品.docx --expected tests/data/1_輸日禽畜肉品_解析結果.json
    python tools/evaluate_breakdown_v2.py --input tests/data/2_歐盟蛋製品問卷.docx --expected tests/data/2_歐盟蛋製品問卷_解析結果.json
    python tools/evaluate_breakdown_v2.py --input tests/data/3_澳洲水產品.md --expected tests/data/3_澳洲水產品_解析結果.json
    python tools/evaluate_breakdown_v2.py --all
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

from app.services.breakdown import (
    _strip_table_of_contents_with_metadata,
    convert_file_to_markdown,
)
from app.services.breakdown_v2 import extract_questions

# ── 預設測試資料集 ────────────────────────────────────────────────────────
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data"
DEFAULT_CASES = [
    ("1_輸日禽畜肉品.docx", "1_輸日禽畜肉品_解析結果.json"),
    ("2_歐盟蛋製品問卷.docx", "2_歐盟蛋製品問卷_解析結果.json"),
    ("3_澳洲水產品.md", "3_澳洲水產品_解析結果.json"),
]


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("人工答案必須是 JSON 陣列")
    return data


def _get_markdown(input_path: Path) -> str:
    """Convert input file to Markdown, handling .md passthrough."""
    if input_path.suffix.lower() == ".md":
        return input_path.read_text(encoding="utf-8")
    return convert_file_to_markdown(str(input_path), input_path.suffix)


async def evaluate(input_path: Path, expected_path: Path) -> dict:
    started_at = time.perf_counter()

    # ── 1. 轉換 Markdown ──
    conversion_started_at = time.perf_counter()
    markdown = _get_markdown(input_path)
    conversion_ms = (time.perf_counter() - conversion_started_at) * 1000
    _, toc_lines_removed, toc_detection_strategy = _strip_table_of_contents_with_metadata(markdown)

    # ── 2. v2 萃取 ──
    extraction_started_at = time.perf_counter()
    actual_items = await extract_questions(markdown)
    extraction_ms = (time.perf_counter() - extraction_started_at) * 1000

    # ── 3. 比對 ──
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
        "extraction_mode": "v2_llm_first",
        "expected_count": len(expected_ids),
        "actual_count": len(actual_ids),
        "true_positive_count": len(true_positives),
        "id_precision": round(precision, 4),
        "id_recall": round(recall, 4),
        "id_f1": round(f1, 4),
        "missing_ids": sorted(expected_ids - actual_ids),
        "extra_ids": sorted(actual_ids - expected_ids),
        "normalized_text_mismatch_ids": text_mismatches,
        "toc_lines_removed": toc_lines_removed,
        "toc_detection_strategy": toc_detection_strategy,
        "conversion_ms": round(conversion_ms, 1),
        "extraction_ms": round(extraction_ms, 1),
        "total_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }


def _print_report(report: dict) -> None:
    """Pretty-print a single evaluation report."""
    name = Path(report["input_file"]).name
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Expected: {report['expected_count']}  |  Actual: {report['actual_count']}  |  TP: {report['true_positive_count']}")
    print(f"  Precision: {report['id_precision']:.4f}  |  Recall: {report['id_recall']:.4f}  |  F1: {report['id_f1']:.4f}")
    if report["missing_ids"]:
        print(f"  Missing IDs ({len(report['missing_ids'])}): {report['missing_ids'][:10]}{'...' if len(report['missing_ids']) > 10 else ''}")
    if report["extra_ids"]:
        print(f"  Extra IDs ({len(report['extra_ids'])}): {report['extra_ids'][:10]}{'...' if len(report['extra_ids']) > 10 else ''}")
    if report["normalized_text_mismatch_ids"]:
        print(f"  Text mismatches ({len(report['normalized_text_mismatch_ids'])}): {report['normalized_text_mismatch_ids'][:10]}")
    print(f"  Conversion: {report['conversion_ms']:.1f} ms  |  Extraction: {report['extraction_ms']:.1f} ms  |  Total: {report['total_ms']:.1f} ms")


async def run_all() -> list[dict]:
    """Run evaluation against all default test cases."""
    reports = []
    for input_name, expected_name in DEFAULT_CASES:
        input_path = TEST_DATA_DIR / input_name
        expected_path = TEST_DATA_DIR / expected_name
        if not input_path.exists() or not expected_path.exists():
            print(f"SKIP: {input_name} (file not found)")
            continue
        print(f"\nEvaluating: {input_name} ...")
        report = await evaluate(input_path, expected_path)
        reports.append(report)
        _print_report(report)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="問卷原始檔案")
    parser.add_argument("--expected", type=Path, help="人工答案 JSON")
    parser.add_argument("--all", action="store_true", help="對所有預設測試案例執行評估")
    parser.add_argument("--min-recall", type=float, default=0.80, help="題號 recall 門檻 (預設 0.80)")
    parser.add_argument("--report", type=Path, help="選填：寫入 JSON 報告的路徑")
    args = parser.parse_args()

    if args.all:
        reports = asyncio.run(run_all())
        if args.report:
            args.report.write_text(
                json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"\n{'=' * 70}")
        print("  SUMMARY")
        print(f"{'=' * 70}")
        all_pass = True
        for report in reports:
            name = Path(report["input_file"]).name
            passed = report["id_recall"] >= args.min_recall
            status = "PASS ✓" if passed else "FAIL ✗"
            print(f"  {status}  {name}  (recall={report['id_recall']:.4f}, threshold={args.min_recall})")
            if not passed:
                all_pass = False
        return 0 if all_pass else 1

    if not args.input or not args.expected:
        parser.error("請提供 --input 和 --expected，或使用 --all")

    report = asyncio.run(evaluate(args.input, args.expected))
    _print_report(report)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(f"\nJSON Report:\n{output}")
    if args.report:
        args.report.write_text(output + "\n", encoding="utf-8")

    return 0 if report["id_recall"] >= args.min_recall else 1


if __name__ == "__main__":
    raise SystemExit(main())
