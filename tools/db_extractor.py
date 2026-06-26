"""
資料庫來源雙語專有名詞擷取腳本

功能：
    從「問卷題目檔」的「題目」與「回覆」欄位（HTML 格式，含中英文）
    提取雙語專有名詞，並將新詞彙整合至「官方正規詞彙」資料表。

處理流程：
    1. 連線 PostgreSQL，查詢「問卷題目檔」WHERE 是否刪除=0
    2. 分別從「題目」與「回覆」欄位提取雙語詞彙：
       a. 解析 HTML，按 <br>/<p>/<div> 等邊界分段
       b. 以中文字符比例分類各段為中文或英文
       c. 合併所有中文段落／英文段落 → 切塊 → LLM 擷取 → 反向驗證
    3. 聚合去重後，將「官方正規詞彙」中不存在的新詞彙寫入資料表

使用方式：
    # 透過環境變數（PG_HOST / PG_PORT / PG_USER / PG_PASSWORD / PG_DB）
    python tools/db_extractor.py

    # 或直接指定連線參數
    python tools/db_extractor.py --host 127.0.0.1 --port 5432 --user postgres --password postgres --db fes

    python tools/db_extractor.py --help
"""

import argparse
import html as html_mod
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser

import psycopg

from aggregator import aggregate_terms, export_csv
from llm_client import create_openai_client, extract_terms_from_qa



# ── 日誌設定 ──────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── HTML 解析 ──────────────────────────────────────────────────────────────────

class _SegmentExtractor(HTMLParser):
    """
    HTML 解析器：在區塊標籤（<br>/<p>/<div> 等）邊界切分文字段落。
    每個段落為一個獨立字串，後續再依中英文比例分類。
    """

    _BLOCK_TAGS = {"br", "p", "div", "li", "tr", "td", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__()
        self._segments: list[str] = []
        self._buffer: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag.lower() in self._BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        # 解碼 HTML 實體（&nbsp; 等），過濾空白行
        text = html_mod.unescape(data).strip()
        if text and text != "\xa0":  # 排除 &nbsp; 殘留
            self._buffer.append(text)

    def _flush(self):
        text = " ".join(self._buffer).strip()
        if text:
            self._segments.append(text)
        self._buffer = []

    def get_segments(self) -> list[str]:
        self._flush()
        return [s for s in self._segments if s]


def _is_chinese_dominant(text: str) -> bool:
    """若文字中 CJK 字符佔比超過 30%，視為中文段落。"""
    if not text:
        return False
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return chinese_chars > len(text) * 0.3


def parse_html_bilingual(html_text: str) -> tuple[str, str]:
    """
    解析包含中英混合內容的 HTML 字串，分離為中文文字與英文文字。

    Args:
        html_text: HTML 格式字串（如題目或回覆欄位內容）

    Returns:
        (zh_text, en_text)：中文段落以 \\n 連接，英文段落以 \\n 連接。
    """
    if not html_text:
        return "", ""

    parser = _SegmentExtractor()
    try:
        parser.feed(html_text)
    except Exception:
        # HTML 解析失敗時，降級為正規表達式去標籤
        import re
        plain = re.sub(r"<[^>]+>", " ", html_text).strip()
        plain = html_mod.unescape(plain).strip()
        if _is_chinese_dominant(plain):
            return plain, ""
        return "", plain

    segments = parser.get_segments()
    zh_parts: list[str] = []
    en_parts: list[str] = []

    for seg in segments:
        if _is_chinese_dominant(seg):
            zh_parts.append(seg)
        else:
            en_parts.append(seg)

    return "\n".join(zh_parts), "\n".join(en_parts)


# ── 資料庫操作 ────────────────────────────────────────────────────────────────

def connect_db(host: str, port: int, user: str, password: str, dbname: str) -> "psycopg.Connection":
    """建立 PostgreSQL 同步連線。"""
    logger = logging.getLogger(__name__)
    logger.info("連線資料庫：%s@%s:%d/%s", user, host, port, dbname)
    # 使用參數化 conninfo 避免特殊字元問題
    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )


def fetch_qa_fields(conn: "psycopg.Connection") -> list[tuple[str, str | None, str | None]]:
    """
    查詢「問卷題目檔」WHERE 是否刪除=0，取得 (主鍵, 題目, 回覆) 列表。
    使用參數化查詢，不拼接 SQL 字串。
    """
    logger = logging.getLogger(__name__)
    with conn.cursor() as cur:
        cur.execute(
            'SELECT "主鍵", "題目", "回覆" FROM public."問卷題目檔" WHERE "是否刪除" = %s',
            (0,),
        )
        rows = cur.fetchall()
    logger.info("查詢「問卷題目檔」共取得 %d 筆資料（是否刪除=0）", len(rows))
    return rows


def load_existing_zh_terms(conn: "psycopg.Connection") -> set[str]:
    """載入「官方正規詞彙」中已有的中文字詞集合，用於去重。"""
    with conn.cursor() as cur:
        cur.execute(
            'SELECT "中文字詞" FROM public."官方正規詞彙" WHERE "中文字詞" IS NOT NULL',
        )
        rows = cur.fetchall()
    return {row[0] for row in rows}


def insert_new_terms(conn: "psycopg.Connection", terms: list[dict]) -> int:
    """
    將新詞彙插入「官方正規詞彙」（僅插入中文字詞不重複的新詞彙）。

    Args:
        conn : psycopg 連線物件
        terms: aggregate_terms() 輸出的詞彙列表，
               每個 dict 含 '中文字詞' 與 '英文字詞' 欄位

    Returns:
        實際插入的筆數。
    """
    logger = logging.getLogger(__name__)
    if not terms:
        return 0

    existing = load_existing_zh_terms(conn)
    logger.info("「官方正規詞彙」現有 %d 筆中文字詞", len(existing))

    new_entries = [
        t for t in terms
        if t.get("中文字詞") and t["中文字詞"] not in existing
    ]

    if not new_entries:
        logger.info("無新詞彙需要寫入")
        return 0

    with conn.cursor() as cur:
        cur.executemany(
            'INSERT INTO public."官方正規詞彙" ("中文字詞", "英文字詞") VALUES (%s, %s)',
            [(t["中文字詞"], t.get("英文字詞", "")) for t in new_entries],
        )
    conn.commit()
    logger.info("成功寫入 %d 筆新詞彙至「官方正規詞彙」", len(new_entries))
    return len(new_entries)


# ── 詞彙擷取 ──────────────────────────────────────────────────────────────────

def extract_from_qa_pairs(
    rows: list[tuple[str, str | None, str | None]],
    client,
    max_workers: int = 5,
) -> list[dict]:
    """
    從問卷題目對（題目 + 回覆）中聯合提取雙語專有名詞。
    採平行呼叫 LLM 以加快執行速度。

    Args:
        rows: (主鍵, 題目, 回覆) 元組列表
        client: OpenAI-compatible 客戶端
        max_workers: 平行執行的工作線程數

    Returns:
        通過擷取的詞彙 list[dict]
    """
    logger = logging.getLogger(__name__)
    logger.info("─" * 60)
    logger.info("開始平行處理「題目 + 回覆」，筆數=%d，線程數=%d", len(rows), max_workers)

    all_terms: list[dict] = []
    
    def _task(pk, title, reply, index, total):
        title_text = title or ""
        reply_text = reply or ""
        if not title_text.strip() and not reply_text.strip():
            return []
        
        logger.info("問卷推論 %d / %d（主鍵=%s）", index, total, pk)
        return extract_terms_from_qa(client, title_text, reply_text)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_task, pk, title, reply, i, len(rows)): pk 
            for i, (pk, title, reply) in enumerate(rows, 1)
        }
        
        for future in as_completed(futures):
            pk = futures[future]
            try:
                terms = future.result()
                all_terms.extend(terms)
                logger.debug("主鍵=%s 擷取 %d 筆詞彙", pk, len(terms))
            except Exception as exc:
                logger.error("主鍵=%s 執行過程發生錯誤：%s", pk, exc)

    logger.info("QA 聯合擷取完成，共 %d 筆詞彙", len(all_terms))
    return all_terms


# ── CLI 參數解析 ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="從「問卷題目檔」提取雙語專有名詞並整合至「官方正規詞彙」",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DB 連線優先順序：命令列參數 > 環境變數（PG_HOST / PG_PORT / PG_USER / PG_PASSWORD / PG_DB）

範例：
  python tools/db_extractor.py
  python tools/db_extractor.py --host 127.0.0.1 --db fes --verbose
  python tools/db_extractor.py --chunk-size 3000 --overlap 300
        """,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("PG_HOST", "127.0.0.1"),
        help="PostgreSQL 主機（預設：PG_HOST 環境變數或 127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PG_PORT", "5432")),
        help="PostgreSQL 埠號（預設：PG_PORT 環境變數或 5432）",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("PG_USER", "postgres"),
        help="PostgreSQL 使用者名稱（預設：PG_USER 環境變數或 postgres）",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("PG_PASSWORD", "postgres"),
        help="PostgreSQL 密碼（預設：PG_PASSWORD 環境變數）",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("PG_DB", "fes"),
        help="資料庫名稱（預設：PG_DB 環境變數或 fes）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="顯示詳細 DEBUG 日誌",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="聚合結果 CSV 輸出路徑（預設：不輸出 CSV）",
    )
    parser.add_argument(
        "--insert_db",
        action="store_true",
        default=False,
        help="將新詞彙寫入資料庫（預設：False）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="平行執行 LLM 的線程數（預設：5）",
    )
    return parser.parse_args()


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    主程式流程：
    1. 連線資料庫
    2. 查詢「問卷題目檔」（是否刪除=0）
    3. 各自從「題目」與「回覆」欄位提取雙語詞彙
    4. 聚合去重
    5. 寫入「官方正規詞彙」（僅新增不存在的詞彙）
    """
    args = _parse_args()
    _setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("═" * 60)
    logger.info("DB 雙語詞彙擷取腳本 啟動")
    logger.info("資料庫：%s@%s:%d/%s", args.user, args.host, args.port, args.db)
    logger.info("═" * 60)

    # ── 初始化 LLM client
    client = create_openai_client()

    # ── 連線資料庫
    try:
        conn = connect_db(args.host, args.port, args.user, args.password, args.db)
    except Exception as exc:
        logger.error("資料庫連線失敗：%s", exc)
        sys.exit(1)

    try:
        # ── 查詢問卷題目檔
        rows = fetch_qa_fields(conn)
        if not rows:
            logger.warning("問卷題目檔查無有效資料，程式結束")
            return

        # ── 聯合擷取（題目 + 回覆）
        all_terms = extract_from_qa_pairs(rows, client, max_workers=args.workers)

        # ── 聚合去重（aggregate_terms 輸出 key 為「中文字詞」/「英文字詞」）
        aggregated = aggregate_terms(all_terms)
        logger.info("聚合後共 %d 組詞彙對", len(aggregated))

        if args.output:
            # ── 匯出 CSV（含負責單位）
            export_csv(aggregated, args.output)

        if args.insert_db:
            # ── 整合寫入「官方正規詞彙」
            inserted = insert_new_terms(conn, aggregated)

        # ── 執行摘要
        logger.info("═" * 60)
        logger.info("【執行摘要】")
        logger.info("  問卷題目筆數  ：%d", len(rows))
        logger.info("  擷取總詞彙    ：%d 筆", len(all_terms))
        logger.info("  聚合後詞彙對  ：%d 組", len(aggregated))
        if args.insert_db:
            logger.info("  寫入官方詞彙  ：%d 筆（新增）", inserted)
        logger.info("═" * 60)

    finally:
        conn.close()
        logger.debug("資料庫連線已關閉")


if __name__ == "__main__":
    main()
