"""
Breakdown service: 問卷檔案 → Markdown → 語義切塊 → 並行 LLM 萃取 → 合併結果。

長文本處理管線架構：
  1. 語義切塊（Semantic Chunking）：依據 Markdown 標題層級拆分，每塊不超過
     _BREAKDOWN_CHUNK_SIZE 字元，確保不溢出本地模型的 Context Window。
  2. 跨區塊狀態繼承（State Injection）：處理第 N 區塊時捕捉最後出現的最淺
     層級父標題；將第 N+1 區塊送入 LLM 時，在 User Message 開頭動態注入該
     標題，讓模型得以正確追溯子項目編號。
  3. Map-Reduce 並行推論：以 asyncio.gather + Semaphore 控制併發上限，
     縮短多區塊處理耗時。
  4. 容錯合併：捕捉單一區塊的 JSONDecodeError / ValueError，跳過失敗區塊，
     僅在全部失敗時才拋出例外，確保部分成功仍能正常回傳。
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

from markitdown import MarkItDown

from app.services.llm import chat_completion

logger = logging.getLogger(__name__)

# ── 切塊與併發參數 ────────────────────────────────────────────────────────
# 單一區塊送入 LLM 的最大字元數。留足餘量給 system prompt（~600 字）
# + 狀態注入（~100 字）+ 模型回應 token（max_tokens=4096）。
_BREAKDOWN_CHUNK_SIZE = 6_000

# LLM 並行呼叫上限，避免同時過多請求打爆 LLM Server。
_MAX_CONCURRENCY = 3

BREAKDOWN_SYSTEM_PROMPT = """\
你是一個精準的資料工程解析器，專門處理非結構化多語系問卷。
你的任務是將輸入的 Markdown 文本解析為扁平化的 JSON 格式，提取出所有的題號與題目內容。

【執行規則】
1. 層級扁平化（Hierarchical Flattening）：
   - 文本中包含多層級標題（如 1.1, 2.2）與子項目（如 (1), (2), a., b., (i), (ii)）。
   - 當遇到子項目時，你必須向前追溯最近的父標題，並將編號合併為點分十進位格式。
   - 範例：父題 "2.2" 下的 "(1)" -> 轉換為 "2.2.1"；父題 "1.2" 下的 "(4)" 下的 "a." -> 轉換為 "1.2.4.a"。
   - 若該項目本身已是完整編號（如 1.1），則直接保留。

2. 語言清洗（Language Purging）：
   - 原始文本混雜了來源國語言（如日文）與英文。你必須完全捨棄任何非英文字元、漢字、假名或來源國語言的描述。
   - 僅提取並保留純英文的題目與敘述。若同一個項目包含雙語，請只萃取英文部分。

3. 雜訊過濾：
   - 忽略問卷的目錄、填寫說明、空白表格標記以及不包含具體問題的純標題區塊。

【輸出約束】
- 必須輸出為合法的 JSON Array of Objects，包含 "question_id" 與 "question_text" 兩個鍵值。
- 絕對禁止在 JSON 前後加上任何解釋性文字或 Markdown 標籤（如 ```json ）。\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  1. 檔案轉 Markdown
# ═══════════════════════════════════════════════════════════════════════════

def _xls_to_markdown(file_path: str) -> str:
    """Convert .xls → Markdown table（pandas + xlrd engine）。"""
    import pandas as pd  # lazy import — pandas is a heavy dep

    sheets = pd.read_excel(file_path, sheet_name=None, engine="xlrd")
    parts: list[str] = []
    for sheet_name, df in sheets.items():
        parts.append(f"## Sheet: {sheet_name}\n")
        parts.append(df.fillna("").to_markdown(index=False))
        parts.append("\n")
    return "\n".join(parts)


def convert_file_to_markdown(file_path: str, file_extension: str) -> str:
    """
    將本地檔案轉換為 Markdown。

    支援：.docx/.doc（mammoth）、.xlsx（openpyxl）、.xls（xlrd）、.pdf（pdfminer）。
    Raises ValueError on conversion failure.
    """
    ext = file_extension.lower().lstrip(".")
    logger.info("檔案轉換開始：路徑=%s  格式=%s", file_path, ext)
    start = time.perf_counter()

    try:
        if ext == "xls":
            markdown = _xls_to_markdown(file_path)
        else:
            md = MarkItDown()
            result = md.convert_local(file_path)
            markdown = result.markdown or ""
    except Exception as e:
        logger.error("檔案轉換失敗：格式=%s  錯誤=%s", ext, e, exc_info=True)
        raise ValueError(f"無法將 .{ext} 檔案轉換為 Markdown：{e}") from e

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "檔案轉換完成：格式=%s  Markdown 長度=%d 字元  耗時=%.1f ms",
        ext, len(markdown), elapsed_ms,
    )
    return markdown


# ═══════════════════════════════════════════════════════════════════════════
#  2. 語義切塊（Semantic Chunking）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarkdownChunk:
    """語義切塊結果。

    Attributes:
        text: 該區塊的 Markdown 文本。
        last_parent_heading: 區塊內最後出現的「最淺層級」標題行
            （例如 ``## 2.2 Standards``）。作為下一區塊的狀態繼承依據：
            前一區塊的 last_parent_heading 會被注入到下一區塊的
            User Message 開頭，讓 LLM 知道子項目應追溯至哪個父標題。
    """
    text: str
    last_parent_heading: str = ""


def _heading_level(heading_line: str) -> int:
    """取得標題層級數（``#`` → 1, ``##`` → 2, …）。非標題回傳 999。"""
    m = re.match(r"^(#{1,6})\s", heading_line)
    return len(m.group(1)) if m else 999


def _find_last_parent_heading(text: str) -> str:
    """
    掃描文本中所有 Markdown 標題行，回傳最後出現的最淺層級標題。

    狀態機邏輯：
      1. 以正則找出所有 ``^#{1,6} `` 標題行
      2. 找出最淺層級（最小 ``#`` 數）
      3. 從後往前搜尋，回傳該層級最後一次出現的標題

    範例：若區塊包含 ``## 2.2 Standards`` → ``### 2.2.1 Water`` → ``### 2.2.2 Pest``
    → 回傳 ``## 2.2 Standards``（level=2 最淺）。
    """
    header_re = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    matches = list(header_re.finditer(text))
    if not matches:
        return ""
    min_level = min(_heading_level(m.group(1)) for m in matches)
    for m in reversed(matches):
        if _heading_level(m.group(1)) == min_level:
            return m.group(1).strip()
    return ""


def split_markdown_semantically(markdown: str) -> list[MarkdownChunk]:
    """
    依據 Markdown 標題層級將長文本拆解為語義區塊。

    切割策略：
      1. 以正則辨識所有標題行（``^#{1,6} …``），作為段落分界
      2. 依序將相鄰段落合併，累積字元數不超過 ``_BREAKDOWN_CHUNK_SIZE``
      3. 超過限制時 flush 為一個 MarkdownChunk
      4. 單一段落本身超過限制時，以換行符做二次切割
      5. 對每個區塊呼叫 ``_find_last_parent_heading()``，記錄該區塊中
         最後出現的最淺層級標題，供下一區塊的狀態繼承使用
      6. 若文本完全不含 Markdown 標題（如純表格），回退為段落切割
    """
    header_re = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    matches = list(header_re.finditer(markdown))

    # ── 無標題回退：以段落（空行）做固定長度切割 ──
    if not matches:
        return _split_plain_text(markdown)

    # ── 拆分為 (heading, body) 段落序列 ──
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        pre = markdown[:matches[0].start()].strip()
        if pre:
            sections.append(("", pre))

    for i, m in enumerate(matches):
        heading = m.group(1)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()
        sections.append((heading, body))

    # ── 合併段落為大小受限的區塊 ──
    chunks: list[MarkdownChunk] = []
    buf: list[str] = []
    buf_len = 0

    def _flush() -> None:
        nonlocal buf, buf_len
        if buf:
            text = "\n\n".join(buf)
            chunks.append(MarkdownChunk(
                text=text,
                last_parent_heading=_find_last_parent_heading(text),
            ))
            buf = []
            buf_len = 0

    for heading, body in sections:
        section_text = f"{heading}\n{body}".strip() if heading else body
        section_len = len(section_text)

        # 單一段落超過限制 → 先 flush 當前 buffer，再二次切割該段落
        if section_len > _BREAKDOWN_CHUNK_SIZE:
            _flush()
            for sub_text in _split_oversized_section(section_text):
                chunks.append(MarkdownChunk(
                    text=sub_text,
                    last_parent_heading=(
                        _find_last_parent_heading(sub_text)
                        or (heading.strip() if heading else "")
                    ),
                ))
            continue

        # 累積字元超過限制 → flush 後再加入新段落
        if buf_len + section_len > _BREAKDOWN_CHUNK_SIZE and buf:
            _flush()

        buf.append(section_text)
        buf_len += section_len

    _flush()
    return chunks or [MarkdownChunk(text=markdown, last_parent_heading="")]


def _split_plain_text(text: str) -> list[MarkdownChunk]:
    """無 Markdown 標題時的回退切割：以段落（連續空行）分界。"""
    if len(text) <= _BREAKDOWN_CHUNK_SIZE:
        return [MarkdownChunk(text=text, last_parent_heading="")]

    chunks: list[MarkdownChunk] = []
    paragraphs = re.split(r"\n{2,}", text)
    buf: list[str] = []
    buf_len = 0

    for para in paragraphs:
        if buf_len + len(para) > _BREAKDOWN_CHUNK_SIZE and buf:
            chunks.append(MarkdownChunk(text="\n\n".join(buf), last_parent_heading=""))
            buf = []
            buf_len = 0
        buf.append(para)
        buf_len += len(para)

    if buf:
        chunks.append(MarkdownChunk(text="\n\n".join(buf), last_parent_heading=""))
    return chunks


def _split_oversized_section(section_text: str) -> list[str]:
    """將超過限制的單一段落以換行符做二次切割。"""
    lines = section_text.split("\n")
    sub_chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for line in lines:
        if buf_len + len(line) + 1 > _BREAKDOWN_CHUNK_SIZE and buf:
            sub_chunks.append("\n".join(buf))
            buf = []
            buf_len = 0
        buf.append(line)
        buf_len += len(line) + 1

    if buf:
        sub_chunks.append("\n".join(buf))
    return sub_chunks


# ═══════════════════════════════════════════════════════════════════════════
#  3. LLM 萃取 + JSON 清洗驗證
# ═══════════════════════════════════════════════════════════════════════════

def _clean_llm_json(raw: str) -> str:
    """剝離 Markdown code fence 與首尾空白。"""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned.strip()


def _validate_breakdown(data: object) -> list[dict]:
    """驗證 JSON 結構：必須為 list[dict]，每筆須含 question_id + question_text。"""
    if not isinstance(data, list):
        raise ValueError(f"LLM 回傳非 JSON Array，實際類型：{type(data).__name__}")
    validated: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 筆資料非物件（dict），實際：{type(item).__name__}")
        if "question_id" not in item or "question_text" not in item:
            raise ValueError(
                f"第 {i} 筆資料缺少必要欄位 question_id / question_text，"
                f"實際鍵值：{list(item.keys())}"
            )
        validated.append({
            "question_id": str(item["question_id"]).strip(),
            "question_text": str(item["question_text"]).strip(),
        })
    return validated


async def _extract_single_chunk(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    state_prefix: str | None = None,
) -> list[dict]:
    """
    對單一區塊執行 LLM 萃取。

    Args:
        chunk_text:   區塊的 Markdown 文本。
        chunk_index:  區塊索引（0-based），僅用於日誌。
        total_chunks: 總區塊數，僅用於日誌。
        state_prefix: 跨區塊狀態繼承前綴（前一區塊的父標題上下文）。
    """
    # 組合 User Message：狀態注入 + Markdown 內容
    parts: list[str] = []
    if state_prefix:
        parts.append(state_prefix)
    parts.append(f"<Markdown>\n{chunk_text}\n</Markdown>")
    user_prompt = "\n\n".join(parts)

    label = f"{chunk_index + 1}/{total_chunks}"
    logger.info(
        "區塊 LLM 萃取開始：%s  提示長度=%d 字元%s",
        label, len(user_prompt),
        "（含狀態注入）" if state_prefix else "",
    )
    start = time.perf_counter()

    raw_output = await chat_completion(
        prompt=user_prompt,
        system_prompt=BREAKDOWN_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=4096,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "區塊 LLM 萃取完成：%s  輸出長度=%d 字元  耗時=%.1f ms",
        label, len(raw_output), elapsed_ms,
    )
    logger.debug("區塊 %s 原始輸出預覽：%s …", label, raw_output[:300])

    # JSON 解析 — 失敗時拋出 ValueError，由上層 gather 捕捉
    cleaned = _clean_llm_json(raw_output)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(
            "區塊 %s JSON 解析失敗：%s\n原始內容前 500 字：%s",
            label, e, cleaned[:500],
        )
        raise ValueError(f"區塊 {label} JSON 解析失敗：{e}") from e

    return _validate_breakdown(parsed)


# ═══════════════════════════════════════════════════════════════════════════
#  4. Map-Reduce 萃取管線（公開介面）
# ═══════════════════════════════════════════════════════════════════════════

async def extract_questions(markdown_text: str) -> list[dict]:
    """
    從 Markdown 文本萃取結構化題號與英文題目。

    管線流程（Map-Reduce with State Injection）：

    **Map 階段**
      1. ``split_markdown_semantically()`` 將 Markdown 依標題層級拆為
         多個 ``MarkdownChunk``，每塊 ≤ ``_BREAKDOWN_CHUNK_SIZE`` 字元。
      2. 為每個區塊建立 LLM 推論任務。若為第 i 塊（i>0），動態注入
         第 i-1 塊的 ``last_parent_heading`` 作為編號追溯上下文。
      3. ``asyncio.gather`` 並行呼叫 LLM，以 ``Semaphore`` 控制最多
         ``_MAX_CONCURRENCY`` 個同時請求。

    **Reduce 階段**
      4. 檢查每個任務的回傳值：成功 → 合併至結果清單；
         失敗（JSONDecodeError / ValueError）→ 記錄警告後跳過。
      5. 若全部區塊均失敗，拋出 ``ValueError`` 通知上層。

    Returns:
        合併後的題目清單 ``[{"question_id": ..., "question_text": ...}, ...]``
    Raises:
        ValueError: 當所有區塊均無法成功萃取時。
    """
    pipeline_start = time.perf_counter()

    # ── Step 1: 語義切塊 ──────────────────────────────────────────────────
    chunks = split_markdown_semantically(markdown_text)
    logger.info(
        "語義切塊完成：Markdown=%d 字元 → %d 個區塊（上限=%d 字元/塊）",
        len(markdown_text), len(chunks), _BREAKDOWN_CHUNK_SIZE,
    )

    # ── Step 2 & 3: Map — 建立並行任務（含跨區塊狀態注入）────────────────
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)
    total = len(chunks)

    async def _guarded_extract(idx: int) -> list[dict]:
        """受 Semaphore 控制的單區塊萃取。"""
        # 狀態繼承：若非首塊，注入前一區塊最後出現的最淺層級父標題
        state_prefix: str | None = None
        if idx > 0:
            prev_heading = chunks[idx - 1].last_parent_heading
            if prev_heading:
                state_prefix = (
                    f"注意：此文本為接續段落。"
                    f"上一段落的最後一個父標題為『{prev_heading}』。"
                    f"請在遇到子項目時，以此標題作為編號追溯的基礎。"
                )
        async with sem:
            return await _extract_single_chunk(
                chunks[idx].text, idx, total, state_prefix,
            )

    tasks = [_guarded_extract(i) for i in range(total)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Step 4: Reduce — 合併成功結果，跳過失敗區塊 ────────────────────────
    merged: list[dict] = []
    success_count = 0
    fail_count = 0

    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            fail_count += 1
            logger.warning(
                "Reduce：區塊 %d/%d 萃取失敗（已跳過）：%s", i + 1, total, result,
            )
        else:
            success_count += 1
            merged.extend(result)

    elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info(
        "Map-Reduce 完成：%d 區塊  成功=%d  失敗=%d  合併題目=%d 筆  耗時=%.1f ms",
        total, success_count, fail_count, len(merged), elapsed_ms,
    )

    # ── Step 5: 全部失敗則拋出例外 ────────────────────────────────────────
    if success_count == 0:
        raise ValueError(
            f"全部 {total} 個區塊均萃取失敗，無法產出任何結果"
        )

    return merged
