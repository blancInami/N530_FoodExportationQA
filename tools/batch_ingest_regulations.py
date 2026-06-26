"""
tools/batch_ingest_regulations.py — 批次匯入法規資料夾

依據子資料夾名稱自動判斷 doc_type，掃描：
  法規/REGULATION/  → doc_type=REGULATION
  法規/GUIDELINE/   → doc_type=GUIDELINE
  法規/QA/          → doc_type=QA

每個檔案的 doc_name 直接取檔名（不含副檔名）。

使用方式：
  python tools/batch_ingest_regulations.py
  python tools/batch_ingest_regulations.py --folder "C:/data/法規"
  python tools/batch_ingest_regulations.py --dry-run
  python tools/batch_ingest_regulations.py --verbose
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# ── 支援的副檔名 ────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md"}

# ── 有效的 doc_type 子資料夾名稱 ────────────────────────────────────────────
VALID_DOC_TYPES: set[str] = {"REGULATION", "GUIDELINE", "QA"}

# ── 預設資料夾（相對於專案根目錄）─────────────────────────────────────────
DEFAULT_FOLDER = Path(__file__).parent.parent / "法規"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批次匯入法規資料夾中的文件至知識文獻庫（doc_type 由子資料夾名稱決定）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
資料夾結構範例：
  法規/
  ├── REGULATION/
  │   ├── 食品安全衛生管理法.pdf
  │   └── 農產品生產及驗證管理法.pdf
  ├── GUIDELINE/
  │   └── 洗選蛋品作業規範.docx
  └── QA/
      └── 輸美問答集.pdf

範例指令：
  python tools/batch_ingest_regulations.py
  python tools/batch_ingest_regulations.py --folder ./法規
  python tools/batch_ingest_regulations.py --dry-run
        """,
    )
    parser.add_argument(
        "--folder",
        default=str(DEFAULT_FOLDER),
        help=f"法規根資料夾路徑（預設：{DEFAULT_FOLDER}）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="傳遞 --verbose 給 ingest_agent（啟用 DEBUG 日誌）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="僅列出將處理的檔案，不實際執行匯入",
    )
    return parser.parse_args()


def collect_files(folder: Path) -> list[tuple[Path, str]]:
    """
    掃描 folder/REGULATION、folder/GUIDELINE、folder/QA 三個子資料夾，
    回傳 (file_path, doc_type) 清單（依 doc_type + 檔名排序）。
    不在上述子資料夾內的檔案會被略過並發出警告。
    """
    results: list[tuple[Path, str]] = []

    for doc_type in sorted(VALID_DOC_TYPES):
        sub = folder / doc_type
        if not sub.is_dir():
            logger.debug("子資料夾不存在，略過：%s", sub)
            continue
        for p in sorted(sub.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                results.append((p, doc_type))

    return results


def ingest_file(file_path: Path, doc_type: str, verbose: bool) -> bool:
    """呼叫 ingest_agent.py 處理單一文件，回傳是否成功。"""
    doc_name = file_path.stem  # 檔名不含副檔名

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "ingest_agent.py"),
        "--input", str(file_path),
        "--doc-name", doc_name,
        "--doc-type", doc_type,
    ]
    if verbose:
        cmd.append("--verbose")

    logger.info("▶ 處理：%s（doc_name=%r, doc_type=%s）", file_path.name, doc_name, doc_type)

    result = subprocess.run(cmd, text=True)

    if result.returncode == 0:
        logger.info("✔ 完成：%s", file_path.name)
        return True
    else:
        logger.error("✘ 失敗：%s（exit code %d）", file_path.name, result.returncode)
        return False


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)

    if not folder.exists():
        logger.error("資料夾不存在：%s", folder)
        sys.exit(1)

    if not folder.is_dir():
        logger.error("路徑不是資料夾：%s", folder)
        sys.exit(1)

    files = collect_files(folder)

    if not files:
        logger.warning("三個子資料夾（REGULATION / GUIDELINE / QA）內均無支援格式的檔案")
        logger.info("支援格式：%s", ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        sys.exit(0)

    logger.info("找到 %d 個文件，資料夾：%s", len(files), folder)

    if args.dry_run:
        logger.info("--- DRY RUN（不實際執行）---")
        for i, (f, dt) in enumerate(files, 1):
            logger.info("  [%d/%d] %-12s %s  →  doc_name=%r", i, len(files), dt, f.name, f.stem)
        sys.exit(0)

    success, failed = 0, []

    for i, (file_path, doc_type) in enumerate(files, 1):
        logger.info("─── [%d/%d] ────────────────────────────────", i, len(files))
        ok = ingest_file(file_path, doc_type, args.verbose)
        if ok:
            success += 1
        else:
            failed.append(f"{doc_type}/{file_path.name}")

    # ── 結果摘要 ────────────────────────────────────────────────────────────
    logger.info("═══ 批次完成：成功 %d / 總計 %d ═══", success, len(files))
    if failed:
        logger.warning("以下 %d 個檔案匯入失敗：", len(failed))
        for name in failed:
            logger.warning("  - %s", name)
        sys.exit(1)


if __name__ == "__main__":
    main()
