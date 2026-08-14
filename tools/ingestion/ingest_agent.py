"""
tools/ingest_agent.py — 知識文獻離線萃取工具

將文件（PDF / DOCX / XLSX 等）結構化後寫入知識文獻三表：
  - 知識文獻主檔
  - 知識文獻節點檔
  - 知識文獻切塊檔（向量切塊）

使用方式：
  python tools/ingest_agent.py --input <file_path> --doc-name <name> [options]

範例：
  python tools/ingest_agent.py \\
      --input "tools/中英法規/food_safety_act.pdf" \\
      --doc-name "食品安全衛生管理法" \\
      --doc-type REGULATION

注意：
  - 本工具完全獨立，不依賴 app/ 任何模組
  - 使用同步 psycopg 而非 async
  - 使用 openai.OpenAI（或 httpx）而非 AsyncClient
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from typing import Any

import httpx
from openai import OpenAI

from db_utils import PLACEHOLDER, get_connection, get_schema

# ── 設定 ────────────────────────────────────────────────────────────────────

# LLM（Gemma 4）
LLM_BASE_URL = os.getenv("TOOLS_LLM_URL", "http://10.166.57.22:40041/v1")
LLM_MODEL = os.getenv("TOOLS_LLM_MODEL", "gemma-4-26B-A4B-it-mtp")
LLM_API_KEY = os.getenv("TOOLS_LLM_API_KEY", "dummy")

# Embedding
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://10.166.57.22:40003/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")

# doc-to-json API（markitdown 降級備援）
DOC2JSON_API_URL = os.getenv("DOC2JSON_API_URL", "http://10.166.57.22:43002/api/doc-to-json/run")

# markitdown 結果低於此字元數視為空，觸發降級
MARKITDOWN_MIN_LENGTH = 50

# 切塊參數
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# LLM 巨觀切片大小（字元）
MACRO_CHUNK_SIZE = 3000

VALID_DOC_TYPES = {"REGULATION", "GUIDELINE", "QA"}

# doc-to-json 降級僅支援 PDF
_PDF_EXTENSIONS = {".pdf"}

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ingest_agent")

# ── LLM System Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一個專業的資料結構化解析引擎。你的任務是閱讀提供的 Markdown 文本，並將其拆解、分類為結構化的 JSON 陣列。

【分類規則 doc_type】
- 若文本具備嚴格的條文（如第X條、第X項），標記為 "REGULATION"。
- 若文本為作業流程、規範指引、SOP，標記為 "GUIDELINE"。
- 若文本為問答形式，標記為 "QA"。

【標題路徑規則 header_path】
- REGULATION：提取章節與條號，如 "第二章 > 第十五條"。
- GUIDELINE：追蹤 Markdown 的標題層級，如 "參、操作原則 > (三)洗淨"。
- QA：提取具體問題，如 "Q1: 申請資格為何？"。

【輸出格式約束】
必須輸出合法且單純的 JSON Array，禁止包含 ```json 標籤或其他任何說明文字。

[
  {
    "doc_type": "REGULATION" | "GUIDELINE" | "QA",
    "header_path": "String (完整的階層路徑)",
    "content": "String (該節點的完整內文，保留原始排版)"
  }
]"""


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def _fallback_doc2json(file_path: str) -> str:
    """
    降級備援：使用 doc-to-json API 將 PDF 轉為純文字。
    僅適用於 PDF 格式；回傳空字串表示 API 也無法取得內容。
    """
    try:
        from pdf_reader import read_pdf  # noqa: PLC0415
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pdf_reader import read_pdf  # noqa: PLC0415

    from pathlib import Path as _Path

    logger.warning("降級觸發：markitdown 失敗，改用 doc-to-json API 讀取：%s", file_path)
    text = read_pdf(_Path(file_path), api_url=DOC2JSON_API_URL)
    if text and text.strip():
        logger.info("doc-to-json 成功取得文字：%d 字元", len(text))
        return text
    logger.error("doc-to-json 也未能取得文字：%s", file_path)
    return ""


def sanitize_text(text: str) -> str:
    """移除 PostgreSQL 不允許的控制字元。"""
    if not text:
        return ""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned.replace("\x00", "").strip()


def sliding_window_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """滑動視窗切塊。"""
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def macro_chunk_markdown(markdown: str) -> list[str]:
    """
    巨觀切片：依 H1/H2 標題邊界切分，單塊超過 MACRO_CHUNK_SIZE 字元時再強制截斷。
    確保每塊都在合理的 Token 範圍內，避免 LLM context 過載。
    """
    # 以 ## 或 # 開頭的行作為分割點
    heading_pattern = re.compile(r"(?m)^#{1,2}\s+")
    splits = list(heading_pattern.finditer(markdown))

    if not splits:
        # 無標題 → 直接按字元上限切
        return [
            markdown[i : i + MACRO_CHUNK_SIZE].strip()
            for i in range(0, len(markdown), MACRO_CHUNK_SIZE)
            if markdown[i : i + MACRO_CHUNK_SIZE].strip()
        ]

    sections: list[str] = []
    boundaries = [m.start() for m in splits] + [len(markdown)]
    for i in range(len(boundaries) - 1):
        section = markdown[boundaries[i] : boundaries[i + 1]].strip()
        if not section:
            continue
        if len(section) <= MACRO_CHUNK_SIZE:
            sections.append(section)
        else:
            # 強制按字元截斷
            for j in range(0, len(section), MACRO_CHUNK_SIZE):
                sub = section[j : j + MACRO_CHUNK_SIZE].strip()
                if sub:
                    sections.append(sub)
    return sections


def _parse_json_response(raw: str) -> list[dict]:
    """
    3 層 JSON fallback 解析：
    1. 直接 json.loads
    2. 擷取第一個 [...] 區塊
    3. 擷取 ```json ... ``` 圍籬內容
    """
    raw = raw.strip()
    # Layer 1
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Layer 2: 擷取第一個 JSON Array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Layer 3: ```json ... ``` 圍籬
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    return []


# ── LLM 呼叫 ─────────────────────────────────────────────────────────────────

def extract_nodes_from_chunk(client: OpenAI, chunk: str) -> list[dict]:
    """
    呼叫 Gemma-4 LLM，將一個 Markdown 切塊結構化為節點清單。
    每個節點包含 doc_type, header_path, content。
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
            temperature=0,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content or ""
        nodes = _parse_json_response(raw)
        # 驗證並過濾非法 doc_type
        valid_nodes = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            dt = node.get("doc_type", "")
            if dt not in VALID_DOC_TYPES:
                logger.warning("節點 doc_type 不合法，略過：%r", dt)
                continue
            if not node.get("header_path") or not node.get("content"):
                logger.warning("節點缺少 header_path 或 content，略過：%r", node)
                continue
            valid_nodes.append(node)
        return valid_nodes
    except Exception as exc:
        logger.error("LLM 呼叫失敗：%s", exc)
        return []


# ── Embedding ────────────────────────────────────────────────────────────────

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    批次向量化，呼叫 Embedding Server（OpenAI compatible）。
    每批最多 16 筆，依 index 排序後回傳。
    """
    if not texts:
        return []

    all_vectors: list[tuple[int, list[float]]] = []
    batch_size = 16

    with httpx.Client(timeout=120.0) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {"model": EMBEDDING_MODEL, "input": batch}
            response = client.post(EMBEDDING_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            for item in data["data"]:
                all_vectors.append((start + item["index"], item["embedding"]))

    all_vectors.sort(key=lambda x: x[0])
    return [v for _, v in all_vectors]


# ── 資料庫寫入 ───────────────────────────────────────────────────────────────

def ingest_to_db(
    doc_name: str,
    doc_type: str,
    file_path: str,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    將萃取結果寫入資料庫：
      1. INSERT 知識文獻主檔（一筆）
      2. INSERT 知識文獻節點檔（多筆）
      3. 對每個節點的 content 切塊 + 向量化 → INSERT 知識文獻切塊檔
    支援 PostgreSQL（psycopg）與 SQL Server（pyodbc）雙後端。
    """
    import os as _os
    _is_mssql = _os.environ.get("DB_TYPE", "postgres").lower() == "mssql"
    schema = get_schema()
    doc_pk = uuid.uuid4().hex[:40]

    # Build dialect-specific chunk INSERT SQL
    if _is_mssql:
        chunk_insert_sql = (
            f'INSERT INTO {schema}."知識文獻切塊檔" ("主鍵", "文獻節點檔主鍵", "切塊內容", "內容向量", "切塊索引", "詞元數量") '
            'VALUES (?, ?, ?, CAST(? AS VECTOR(1024)), ?, ?)'
        )
    else:
        chunk_insert_sql = (
            f'INSERT INTO {schema}."知識文獻切塊檔" ("主鍵", "文獻節點檔主鍵", "切塊內容", "內容向量", "切塊索引", "詞元數量") '
            'VALUES (%s, %s, %s, %s::vector, %s, %s)'
        )

    with get_connection() as conn:
        with conn.cursor() as cur:
            # ── 寫入知識文獻主檔 ────────────────────────────────────────────
            cur.execute(
                f'INSERT INTO {schema}."知識文獻主檔" ("主鍵", "文獻名稱", "文獻類型", "原始檔案路徑") '
                f'VALUES ({PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER})',
                (doc_pk, sanitize_text(doc_name), doc_type, sanitize_text(file_path)),
            )
            logger.info("知識文獻主檔寫入：pk=%s  名稱=%s  類型=%s", doc_pk, doc_name, doc_type)

            # ── 寫入知識文獻節點檔 ──────────────────────────────────────────────
            node_rows: list[tuple] = []
            node_pks: list[str] = []
            for idx, node in enumerate(nodes, start=1):
                node_pk = uuid.uuid4().hex[:40]
                node_pks.append(node_pk)
                node_rows.append((
                    node_pk,
                    doc_pk,
                    sanitize_text(node["header_path"]),
                    sanitize_text(node["content"]),
                    idx,
                ))
            cur.executemany(
                f'INSERT INTO {schema}."知識文獻節點檔" ("主鍵", "文獻主檔主鍵", "節點標題路徑", "節點內容", "排序索引") '
                f'VALUES ({PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER})',
                node_rows,
            )
            logger.info("知識文獻節點檔寫入：%d 筆", len(node_rows))

            # ── 切塊 + 向量化 + 寫入知識文獻切塊檔 ────────────────────────────
            total_chunks = 0
            for node_pk, node in zip(node_pks, nodes):
                content = node["content"]
                chunks = sliding_window_chunk(content)
                if not chunks:
                    continue

                header_path = sanitize_text(node["header_path"])
                # 避免 LLM 幻覺產出超長 header_path 導致 Embedding 413 錯誤
                if len(header_path) > 300:
                    header_path = header_path[:300] + "..."

                sanitized = []
                for c in chunks:
                    clean_text = sanitize_text(c)
                    if header_path and header_path not in clean_text:
                        clean_text = f"[{header_path}]\n{clean_text}"
                    sanitized.append(clean_text)

                texts_for_embed = [t if t else "empty" for t in sanitized]
                logger.info("向量化節點：節點主鍵=%s  切塊數=%d", node_pk, len(texts_for_embed))
                vectors = get_embeddings_batch(texts_for_embed)

                chunk_rows: list[tuple] = []
                for chunk_idx, (chunk_text, vector) in enumerate(zip(sanitized, vectors)):
                    token_count = len(chunk_text.encode("utf-8")) // 4  # 粗估
                    vector_str = "[" + ",".join(str(v) for v in vector) + "]"
                    chunk_rows.append((
                        str(uuid.uuid4()),
                        node_pk,
                        chunk_text,
                        vector_str,
                        chunk_idx,
                        token_count,
                    ))

                cur.executemany(chunk_insert_sql, chunk_rows)
                total_chunks += len(chunk_rows)

    logger.info(
        "匯入完成：文獻主鍵=%s  節點數=%d  切塊數=%d",
        doc_pk, len(nodes), total_chunks,
    )
    return {
        "status": "success",
        "文獻主鍵": doc_pk,
        "節點數": len(nodes),
        "切塊數": total_chunks,
    }


# ── CLI 主程式 ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="知識文獻離線萃取工具：將文件結構化後寫入知識文獻庫",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 法規 PDF
  python tools/ingest_agent.py --input law.pdf --doc-name "食品安全衛生管理法"

  # 作業指引 DOCX（手動指定類型）
  python tools/ingest_agent.py --input sop.docx --doc-name "洗選蛋品作業規範" --doc-type GUIDELINE

  # 問答集 PDF
  python tools/ingest_agent.py --input faq.pdf --doc-name "輸美問答集" --doc-type QA

  # 使用純 Markdown 輸入
  python tools/ingest_agent.py --input content.md --doc-name "食品安全法" --is-markdown
        """,
    )
    parser.add_argument("--input", required=True, help="輸入文件路徑（PDF / DOCX / XLSX / MD）")
    parser.add_argument("--doc-name", required=True, help="文獻名稱（寫入 知識文獻主檔.文獻名稱）")
    parser.add_argument(
        "--doc-type",
        choices=list(VALID_DOC_TYPES),
        default=None,
        help="文獻類型（REGULATION / GUIDELINE / QA）。若不指定，由 LLM 自動判斷每個節點的類型",
    )
    parser.add_argument(
        "--is-markdown",
        action="store_true",
        help="若輸入已是 Markdown 格式，跳過 markitdown 轉換",
    )
    parser.add_argument("--verbose", action="store_true", help="啟用 DEBUG 層級日誌")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Step 1: 讀取 / 轉換 Markdown ────────────────────────────────────────
    used_doc2json = False
    input_ext = os.path.splitext(args.input)[1].lower()

    if args.is_markdown:
        with open(args.input, encoding="utf-8") as f:
            markdown = f.read()
        logger.info("直接載入 Markdown：%d 字元", len(markdown))
    else:
        logger.info("轉換文件為 Markdown：%s", args.input)
        markdown = ""
        try:
            try:
                from tools.markdown_converter import convert_to_markdown  # noqa: PLC0415
            except ImportError:
                # 允許直接執行腳本時以相對路徑匯入
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from tools.markdown_converter import convert_to_markdown  # noqa: PLC0415
            markdown = convert_to_markdown(args.input)
        except Exception as exc:
            logger.warning("markitdown 轉換失敗：%s", exc)

        # markitdown 結果為空或不足閾值 → 觸發降級 A
        if len(markdown.strip()) < MARKITDOWN_MIN_LENGTH:
            if input_ext in _PDF_EXTENSIONS:
                logger.warning(
                    "markitdown 結果不足 %d 字元（實際 %d），觸發 doc-to-json 降級",
                    MARKITDOWN_MIN_LENGTH, len(markdown.strip()),
                )
                markdown = _fallback_doc2json(args.input)
                used_doc2json = True
                if not markdown.strip():
                    logger.error("markitdown 與 doc-to-json 均無法取得文字，中止")
                    sys.exit(1)
            else:
                logger.error(
                    "markitdown 轉換失敗且副檔名 %s 不支援 doc-to-json 降級，中止", input_ext
                )
                sys.exit(1)

        logger.info(
            "文本取得完成：%d 字元%s",
            len(markdown),
            "（via doc-to-json）" if used_doc2json else "",
        )

    # ── Step 2: 巨觀切片 ────────────────────────────────────────────────────
    macro_chunks = macro_chunk_markdown(markdown)
    logger.info("巨觀切片完成：%d 塊", len(macro_chunks))

    # ── Step 3: LLM 結構化萃取 ──────────────────────────────────────────────
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    all_nodes: list[dict] = []
    for i, chunk in enumerate(macro_chunks, start=1):
        logger.info("LLM 萃取：切塊 [%d/%d]  長度=%d", i, len(macro_chunks), len(chunk))
        nodes = extract_nodes_from_chunk(client, chunk)
        # 若 CLI 指定了 doc_type，強制覆蓋 LLM 判斷結果
        if args.doc_type:
            for node in nodes:
                node["doc_type"] = args.doc_type
        all_nodes.extend(nodes)
        logger.info("切塊 [%d/%d] 萃取到 %d 個節點", i, len(macro_chunks), len(nodes))

    if not all_nodes:
        if used_doc2json or input_ext not in _PDF_EXTENSIONS:
            # 已試過 doc-to-json，或檔案格式不支援降級 → 直接中止
            logger.error("LLM 未萃取到任何節點，中止")
            sys.exit(1)

        # 降級 B：markitdown 產出品質不佳導致 LLM 0 節點，改用 doc-to-json 重試
        logger.warning("LLM 未從 markitdown 結果萃取到節點，觸發 doc-to-json 降級重試")
        fallback_text = _fallback_doc2json(args.input)
        if not fallback_text.strip():
            logger.error("doc-to-json 也未能取得文字，中止")
            sys.exit(1)

        used_doc2json = True
        macro_chunks = macro_chunk_markdown(fallback_text)
        logger.info("降級重試：巨觀切片 %d 塊", len(macro_chunks))
        for i, chunk in enumerate(macro_chunks, start=1):
            logger.info("降級 LLM 萃取：切塊 [%d/%d]  長度=%d", i, len(macro_chunks), len(chunk))
            nodes = extract_nodes_from_chunk(client, chunk)
            if args.doc_type:
                for node in nodes:
                    node["doc_type"] = args.doc_type
            all_nodes.extend(nodes)
            logger.info("降級切塊 [%d/%d] 萃取到 %d 個節點", i, len(macro_chunks), len(nodes))

        if not all_nodes:
            logger.error("降級重試後 LLM 仍未萃取到任何節點，中止")
            sys.exit(1)

    logger.info("總節點數：%d", len(all_nodes))

    # 推斷文獻整體的 doc_type（取最多票的類型，用於 主檔記錄）
    if args.doc_type:
        final_doc_type = args.doc_type
    else:
        type_counts: dict[str, int] = {}
        for node in all_nodes:
            dt = node.get("doc_type", "REGULATION")
            type_counts[dt] = type_counts.get(dt, 0) + 1
        final_doc_type = max(type_counts, key=lambda k: type_counts[k])
        logger.info("自動推斷文獻類型：%s（票數分布：%s）", final_doc_type, type_counts)

    # ── Step 4: 寫入資料庫 ──────────────────────────────────────────────────
    result = ingest_to_db(
        doc_name=args.doc_name,
        doc_type=final_doc_type,
        file_path=args.input,
        nodes=all_nodes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
