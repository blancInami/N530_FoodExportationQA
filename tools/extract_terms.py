"""
雙語專有名詞擷取主程式（CLI 入口）

功能概述：
    走訪指定資料夾中的中英文配對 PDF，
    透過本地 Gemma 4 模型擷取雙語專有名詞，
    以反向驗證機制過濾幻覺詞彙，
    最終聚合統計並匯出為 CSV。

完整處理管線：
    1. 初始化 OpenAI client（指向本地 Gemma 4）
    2. 走訪資料夾，配對中英文 PDF
    3. 逐對處理：
       a. 讀取中文 PDF（doc-to-json API）
       b. 讀取英文 PDF（doc-to-json API，與中文流程相同）
       c. 中英文文本各自切塊
       d. 比例索引對齊，逐對呼叫 LLM 擷取專有名詞
       e. 反向驗證：過濾未出現在全文中的幻覺詞彙
    4. 全域聚合所有通過驗證的詞彙對
    5. 匯出 CSV（頻率降冪排序）

使用方式：
    python tools/extract_terms.py --input-dir ./pdfs --output terms.csv

    更多選項：
    python tools/extract_terms.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

from aggregator import aggregate_terms, export_csv
from chunking import align_chunks, chunk_text
from file_pairing import discover_pdf_pairs
from llm_client import create_openai_client, extract_terms_from_chunk
from pdf_reader import read_pdf
from validator import filter_hallucinations

# ── 日誌設定 ──────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    """
    設定 logging 格式與等級。
    verbose=True 時顯示 DEBUG 訊息，否則只顯示 INFO 及以上。
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── 單一 PDF 對處理 ────────────────────────────────────────────────────────────

def process_pdf_pair(
    zh_path: Path,
    en_path: Path,
    client,
    api_url: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    處理一對中英文 PDF，回傳通過反向驗證的詞彙列表。

    此函式封裝了完整的單對處理流程，以便主程式的 try/except
    能針對每一對 PDF 獨立捕捉錯誤，防止單一檔案失敗中斷整體。

    Args:
        zh_path   : 中文 PDF 路徑
        en_path   : 英文 PDF 路徑
        client    : OpenAI-compatible 客戶端（指向本地 Gemma 4）
        api_url   : doc-to-json API 的完整 URL
        chunk_size: 分塊大小（字元數）
        overlap   : 相鄰分塊重疊字元數

    Returns:
        通過反向驗證的詞彙 list[dict]。
        若讀取或推論失敗，回傳空列表。
    """
    logger = logging.getLogger(__name__)
    logger.info("─" * 60)
    logger.info("開始處理：中文=%s  英文=%s", zh_path.name, en_path.name)

    # ── 步驟 a：透過 doc-to-json API 讀取 PDF 文字 ────────────────────────
    zh_text = read_pdf(zh_path, api_url=api_url)
    en_text = read_pdf(en_path, api_url=api_url)

    # 若任一文件讀取結果為空，跳過此配對
    if not zh_text.strip():
        logger.warning("中文 PDF 萃取結果為空，跳過此配對：%s", zh_path.name)
        return []
    if not en_text.strip():
        logger.warning("英文 PDF 萃取結果為空，跳過此配對：%s", en_path.name)
        return []

    logger.info(
        "文本萃取完成：中文 %d 字元  英文 %d 字元",
        len(zh_text), len(en_text),
    )

    # ── 步驟 b：分塊 ───────────────────────────────────────────────────────
    zh_chunks = chunk_text(zh_text, chunk_size=chunk_size, overlap=overlap)
    en_chunks = chunk_text(en_text, chunk_size=chunk_size, overlap=overlap)

    logger.info("分塊完成：中文 %d 塊  英文 %d 塊", len(zh_chunks), len(en_chunks))

    # ── 步驟 c：比例索引對齊 ───────────────────────────────────────────────
    chunk_pairs = align_chunks(zh_chunks, en_chunks)

    if not chunk_pairs:
        logger.warning("Chunk 對齊結果為空，跳過此配對")
        return []

    # ── 步驟 d：逐 chunk 呼叫 LLM 擷取專有名詞 ────────────────────────────
    raw_terms: list[dict] = []
    for idx, (zh_chunk, en_chunk) in enumerate(chunk_pairs, start=1):
        logger.info("推論 Chunk %d / %d ...", idx, len(chunk_pairs))
        chunk_result = extract_terms_from_chunk(client, zh_chunk, en_chunk)
        raw_terms.extend(chunk_result)
        logger.debug("Chunk %d 擷取到 %d 筆原始詞彙", idx, len(chunk_result))

    logger.info("LLM 推論完成：共擷取 %d 筆原始詞彙（含可能幻覺）", len(raw_terms))

    # ── 步驟 e：反向驗證，過濾幻覺詞彙 ───────────────────────────────────
    # 注意：使用完整全文（zh_text / en_text）而非 chunk，避免邊界截斷誤判
    validated_terms = filter_hallucinations(raw_terms, zh_text, en_text)

    logger.info(
        "處理完成：%s + %s  → 驗證通過 %d 筆",
        zh_path.name, en_path.name, len(validated_terms),
    )
    return validated_terms


# ── CLI 參數解析 ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """解析 CLI 參數並回傳 Namespace 物件。"""
    parser = argparse.ArgumentParser(
        description="雙語專有名詞擷取腳本（防幻覺機制）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python tools/extract_terms.py --input-dir ./pdfs --output terms.csv
  python tools/extract_terms.py --input-dir ./pdfs --chunk-size 3000 --overlap 300 --verbose
        """,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="包含中英文配對 PDF 的資料夾路徑（必要）",
    )
    parser.add_argument(
        "--output",
        default="terms_output.csv",
        metavar="FILE",
        help="CSV 輸出路徑（預設：terms_output.csv）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        metavar="N",
        help="每個文本切塊的最大字元數（預設：2000）",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=300,
        metavar="N",
        help="相鄰切塊的重疊字元數（預設：300）",
    )
    parser.add_argument(
        "--api-url",
        default="http://10.88.91.72:43002/api/doc-to-json/run",
        metavar="URL",
        help="doc-to-json API 端點（預設：http://10.88.91.72:43002/api/doc-to-json/run）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="顯示詳細 DEBUG 日誌",
    )

    return parser.parse_args()


# ── 主程式 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    主程式入口。

    流程：
    1. 解析 CLI 參數
    2. 初始化 OpenAI client（指向本地 Gemma 4）
    3. 走訪資料夾，發現 PDF 配對
    4. 逐對處理（每對獨立 try/except，防止單一失敗中斷全程）
    5. 全域聚合通過驗證的詞彙
    6. 匯出 CSV
    7. 印出執行摘要
    """
    args = _parse_args()
    _setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("═" * 60)
    logger.info("雙語專有名詞擷取腳本 啟動")
    logger.info("輸入資料夾：%s", args.input_dir)
    logger.info("輸出 CSV  ：%s", args.output)
    logger.info("Chunk 大小：%d  重疊：%d", args.chunk_size, args.overlap)
    logger.info("API URL   ：%s", args.api_url)
    logger.info("═" * 60)

    # ── 步驟 1：初始化 OpenAI client ─────────────────────────────────────
    client = create_openai_client()

    # ── 步驟 2：發現 PDF 配對 ─────────────────────────────────────────────
    try:
        pdf_pairs = discover_pdf_pairs(args.input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("資料夾錯誤：%s", exc)
        sys.exit(1)

    if not pdf_pairs:
        logger.warning("未找到任何中英文 PDF 配對，程式結束")
        sys.exit(0)

    logger.info("共找到 %d 組配對，開始處理...", len(pdf_pairs))

    # ── 步驟 3：逐對處理 ─────────────────────────────────────────────────
    all_validated_terms: list[dict] = []
    success_count = 0
    fail_count = 0
    total_raw = 0
    total_validated = 0

    for zh_path, en_path in pdf_pairs:
        try:
            pair_terms = process_pdf_pair(
                zh_path=zh_path,
                en_path=en_path,
                client=client,
                api_url=args.api_url,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
            all_validated_terms.extend(pair_terms)
            total_validated += len(pair_terms)
            success_count += 1

        except Exception as exc:
            # 捕捉所有未預期的例外，防止單一檔案對失敗中斷整體流程
            logger.error(
                "處理失敗（跳過此配對）：中文=%s  英文=%s  錯誤：%s",
                zh_path.name, en_path.name, exc,
            )
            fail_count += 1

    # ── 步驟 4：全域聚合 ──────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("開始全域聚合...")
    aggregated = aggregate_terms(all_validated_terms)

    # ── 步驟 5：匯出 CSV ──────────────────────────────────────────────────
    export_csv(aggregated, args.output)

    # ── 步驟 6：印出執行摘要 ──────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("【執行摘要】")
    logger.info("  處理配對總數：%d", len(pdf_pairs))
    logger.info("  成功 / 失敗 ：%d / %d", success_count, fail_count)
    logger.info("  通過驗證詞彙：%d 筆（含重複）", total_validated)
    logger.info("  去重後詞彙對：%d 組", len(aggregated))
    logger.info("  CSV 輸出路徑：%s", Path(args.output).resolve())
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
