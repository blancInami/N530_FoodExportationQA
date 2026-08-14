"""
Breakdown v2: LLM-First 問卷解析管線（方案 A：全域骨架驅動 + 祖先路徑並行萃取）。

以 LLM 為核心，結合全域章節骨架掃描（Pass 1）與帶祖先路徑的並行萃取（Pass 2），
徹底解決長文本切塊時主章節丟失與多層級編號斷鏈問題。

管線架構：
  1. 前處理：TOC 移除 + NFKC 正規化
  2. Pass 1（全域骨架掃描）：
     從全文提取潛在標題行，由 LLM 快速建構「全域章節大綱樹（Document Outline Tree）」。
  3. 語義切塊與祖先路徑綁定：
     依 Markdown 標題切塊，並將每個 Chunk 與全域大綱樹對齊，賦予該 Chunk 明確的
     「當前祖先路徑棧（Active Ancestor Path）」與「規範 ID 前綴（Canonical Prefix）」。
  4. Pass 2（帶路徑並行萃取）：
     注入 <Active_Ancestor_Path> 與 <Outline_Context>，LLM 萃取題目時強制接續該路徑。
  5. Pass 3（Reduce 與全域一致性審核）：
     合併結果、去重、並進行全局一致性檢驗。
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.services.llm import chat_completion
from app.services.breakdown import (
    convert_file_to_markdown,
    strip_table_of_contents,
)
from app.services.breakdown_preprocessing import normalize_markdown_for_breakdown

logger = logging.getLogger(__name__)

# ── 切塊與併發參數 ────────────────────────────────────────────────────────
_BREAKDOWN_CHUNK_SIZE = 6_000
_CONSISTENCY_MAX_ITEMS = 120
_MAX_RETRIES = 1


# ═══════════════════════════════════════════════════════════════════════════
#  LLM System Prompts
# ═══════════════════════════════════════════════════════════════════════════

SKELETON_OUTLINE_SYSTEM_PROMPT = """\
你是一個精準的文件大綱結構分析專家。
你的任務是根據輸入的文件標題候選行，梳理並建立整份問卷的「規範章節大綱樹（Document Outline Tree）」。

【執行規則】
1. 階層辨識：
   - 識別所有章節標題（如 Part A, Chapter I, 1. General, 2.2 Standards 等）。
   - 判斷各標題之間的父子層級關係（如 Part A -> 1. -> 1.1）。
2. 編號標準化（Canonical ID）：
   - 為每個章節/標題節點賦予一個規範的層級編號（如 "I", "I.1", "I.1.1", "Part_A.1"）。
3. 雜訊過濾：
   - 忽略純表格內容、具體題目描述、填寫指示等非章節標題。

【輸出約束】
- 輸出合法的 JSON Array of Objects，每個物件代表一個章節標題節點：
  [
    {
      "canonical_id": "規範層級編號 (如 I, I.1, 2, 2.1)",
      "heading_title": "標題名稱 (純英文或主要標題)",
      "raw_marker": "原始標題在文本中的標記 (如 ## 1.1 或 **Chapter I**)"
    }
  ]
- 必須按照在原文中出現的順序排列。
- 絕對禁止在 JSON 前後加上任何解釋性文字或 Markdown 標籤（如 ```json ）。\
"""

EXTRACTION_SYSTEM_PROMPT = """\
你是一個精準的資料工程解析器，專門處理來自各國的非結構化多語系問卷。
你的任務是將輸入的 Markdown 文本解析為扁平化的 JSON 格式，提取出所有的題號與題目內容。

【執行規則】
1. 祖先路徑繼承（Ancestor Path Inheritance）- 極重要：
   - 若提供了 <Active_Ancestor_Path>，表示當前文本片段處於該主章節階層之下。
   - 所有子題目與子項目的 question_id 必須以 <Active_Ancestor_Path> 中指定的編號前綴為基底展開，不得遺漏主章節編號！
   - 範例：若 Active_Ancestor_Path 為 "II.1"，當前文本中出現 "(1)"，其 ID 必須輸出為 "II.1.1"（或點分十進位）；出現 "a." 則為 "II.1.1.a"。

2. 編號辨識與層級扁平化：
   - 辨識文本中各種編號格式（點分十進位 1.1, 羅馬數字 I/II, 括號數字 (1)/(2), 字母 a/b/c, 羅馬子項 (i)/(ii) 等）。
   - 向前追溯本片段或 Ancestor Path 的父標題，將編號合併為點分格式。
   - 字母項目與羅馬數字子項統一轉為小寫（如 "a", "b", "i", "ii"）。

3. 語言清洗（Language Purging）：
   - 原始文本可能混雜來源國語言（如日文、中文、韓文、泰文等）與英文。
   - 僅提取並保留純英文的題目與敘述。若同一個項目包含雙語，只萃取英文部分。
   - 完全捨棄任何非英文的描述性文字。

4. 雜訊過濾：
   - 忽略問卷的目錄、填寫說明、空白表格標記、純裝飾性標題。
   - 不包含具體問題的純標題區塊不應產生獨立條目。
   - 表格的欄位標題行（如 "No. | Detailed Questions | ..."）不是題目，請忽略。

【輸出約束】
- 輸出合法的 JSON Array of Objects，每個物件包含 "question_id" 與 "question_text" 兩個鍵值。
- question_id 必須是字串格式的完整扁平化編號（包含主章節與子項目編號）。
- question_text 必須是純英文文本，不含題號前綴。
- 絕對禁止在 JSON 前後加上任何解釋性文字或 Markdown 標籤（如 ```json ）。\
"""

CONSISTENCY_SYSTEM_PROMPT = """\
你是資料品質檢核員。你收到的是從一份問卷的多個段落分別萃取後合併的題目清單。
你的任務是檢核並修正可能的問題，提升整體資料品質。

【檢核規則】
1. 重複合併：若有相同 question_id 的多筆條目，合併其 question_text（以空格連接）。
2. 主章節與層級完整性：檢查題號前綴是否完整，修正跨切塊時可能產生的層級斷鏈或格式混用。
3. 移除雜訊：刪除明顯不是問題的條目（如只有標題沒有實質問題內容的純容器項目）。
4. 穩定性原則：
   - 不要修改已經正確的題號與內容。
   - 不要新增原清單中不存在的題目。
   - 不要重新編號——保留原始編號系統。
   - 若無法確定某條目是否應刪除，保留它。

【輸出】
修正後的 JSON Array of Objects，每個包含 "question_id" 與 "question_text"。
不得加入任何解釋性文字或 Markdown 標籤。\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  1. Pass 1: 全域章節骨架掃描 (Document Skeleton Extraction)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OutlineNode:
    """全域章節大綱節點。"""
    canonical_id: str
    heading_title: str
    raw_marker: str = ""
    source_position: int = 0


def _extract_heading_candidates(markdown_text: str) -> list[tuple[int, str]]:
    """提取全文中潛在的標題行及其在全文中的字元偏移量。"""
    candidates: list[tuple[int, str]] = []
    # 匹配 Markdown headings (#..######), 粗體獨立行 (**...**), 羅馬章節, Part/Section 開頭
    pattern = re.compile(
        r"^(?:#{1,6}\s+.+|\*\*[^*]+\*\*|(?:Part|Section|Chapter|Annex|Appendix)\s+[A-Z0-9IVXLCDM]+.*)$",
        re.MULTILINE | re.IGNORECASE,
    )
    for match in pattern.finditer(markdown_text):
        line = match.group(0).strip()
        # 過濾過長的普通段落（超過 150 字元通常不是純標題）
        if len(line) <= 150:
            candidates.append((match.start(), line))
    return candidates


async def _extract_document_skeleton(markdown_text: str) -> list[OutlineNode]:
    """Pass 1: 呼叫 LLM 從全文標題候選行建立全域大綱結構樹。"""
    candidates = _extract_heading_candidates(markdown_text)
    if not candidates:
        logger.info("v2 Pass 1: 未檢測到顯式標題候選行，跳過全域骨架掃描")
        return []

    outline_input = "\n".join(f"- [pos:{pos}] {line}" for pos, line in candidates)
    # 限制提示長度，避免大綱提示本身過大
    if len(outline_input) > 12_000:
        outline_input = outline_input[:12_000] + "\n...(truncated)"

    logger.info("v2 Pass 1: 全域大綱掃描開始，候選行數=%d", len(candidates))
    start = time.perf_counter()

    raw_output = await chat_completion(
        prompt=f"<Heading_Candidates>\n{outline_input}\n</Heading_Candidates>",
        system_prompt=SKELETON_OUTLINE_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=4096,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("v2 Pass 1: 全域大綱掃描完成，耗時=%.1f ms", elapsed_ms)

    cleaned = _clean_llm_json(raw_output)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, list):
            return []
        nodes: list[OutlineNode] = []
        for item in data:
            if isinstance(item, dict) and "canonical_id" in item:
                raw_marker = str(item.get("raw_marker", "")).strip()
                # 尋找最近的位置偏移
                matched_pos = 0
                for pos, cand_line in candidates:
                    if raw_marker and raw_marker in cand_line:
                        matched_pos = pos
                        break
                nodes.append(OutlineNode(
                    canonical_id=str(item["canonical_id"]).strip(),
                    heading_title=str(item.get("heading_title", "")).strip(),
                    raw_marker=raw_marker,
                    source_position=matched_pos,
                ))
        logger.info("v2 Pass 1: 成功解析全域大綱節點=%d 個", len(nodes))
        return nodes
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("v2 Pass 1: 全域大綱 JSON 解析失敗，回退至局部上下文：%s", e)
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  2. 語義切塊與祖先路徑綁定 (Semantic Chunking & Path Binding)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarkdownChunk:
    """語義切塊結果（方案 A：包含祖先路徑與全域大綱上下文）。"""
    text: str
    start_pos: int = 0
    end_pos: int = 0
    last_parent_heading: str = ""
    active_ancestor_path: str = ""
    active_ancestor_title: str = ""


def _heading_level(heading_line: str) -> int:
    """取得標題層級數（``#`` → 1, ``##`` → 2, …）。非標題回傳 999。"""
    m = re.match(r"^(#{1,6})\s", heading_line)
    return len(m.group(1)) if m else 999


def _find_last_parent_heading(text: str) -> str:
    """掃描文本中所有 Markdown 標題行，回傳最後出現的最淺層級標題。"""
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
    """依據 Markdown 標題層級將長文本拆解為語義區塊，並記錄精準字元偏移位置。"""
    header_re = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    matches = list(header_re.finditer(markdown))

    if not matches:
        return _split_plain_text(markdown)

    sections: list[tuple[str, str, int, int]] = []
    if matches[0].start() > 0:
        pre = markdown[:matches[0].start()].strip()
        if pre:
            sections.append(("", pre, 0, matches[0].start()))

    for i, m in enumerate(matches):
        heading = m.group(1)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()
        sections.append((heading, body, m.start(), body_end))

    chunks: list[MarkdownChunk] = []
    buf: list[str] = []
    buf_len = 0
    chunk_start = 0
    chunk_end = 0

    def _flush() -> None:
        nonlocal buf, buf_len, chunk_start, chunk_end
        if buf:
            text = "\n\n".join(buf)
            chunks.append(MarkdownChunk(
                text=text,
                start_pos=chunk_start,
                end_pos=chunk_end,
                last_parent_heading=_find_last_parent_heading(text),
            ))
            buf = []
            buf_len = 0

    for heading, body, s_start, s_end in sections:
        section_text = f"{heading}\n{body}".strip() if heading else body
        section_len = len(section_text)

        if section_len > _BREAKDOWN_CHUNK_SIZE:
            _flush()
            sub_chunks = _split_oversized_section(section_text)
            curr_pos = s_start
            for sub_text in sub_chunks:
                sub_end = curr_pos + len(sub_text)
                chunks.append(MarkdownChunk(
                    text=sub_text,
                    start_pos=curr_pos,
                    end_pos=sub_end,
                    last_parent_heading=(
                        _find_last_parent_heading(sub_text)
                        or (heading.strip() if heading else "")
                    ),
                ))
                curr_pos = sub_end
            continue

        if buf_len + section_len > _BREAKDOWN_CHUNK_SIZE and buf:
            _flush()
            chunk_start = s_start

        if not buf:
            chunk_start = s_start
        buf.append(section_text)
        buf_len += section_len
        chunk_end = s_end

    _flush()
    return chunks or [MarkdownChunk(text=markdown, start_pos=0, end_pos=len(markdown))]


def _split_plain_text(text: str) -> list[MarkdownChunk]:
    """無 Markdown 標題時的回退切割。"""
    if len(text) <= _BREAKDOWN_CHUNK_SIZE:
        return [MarkdownChunk(text=text, start_pos=0, end_pos=len(text))]

    chunks: list[MarkdownChunk] = []
    paragraphs = re.split(r"\n{2,}", text)
    buf: list[str] = []
    buf_len = 0
    curr_start = 0

    for para in paragraphs:
        if buf_len + len(para) > _BREAKDOWN_CHUNK_SIZE and buf:
            chunk_str = "\n\n".join(buf)
            chunks.append(MarkdownChunk(
                text=chunk_str,
                start_pos=curr_start,
                end_pos=curr_start + len(chunk_str),
            ))
            curr_start += len(chunk_str)
            buf = []
            buf_len = 0
        buf.append(para)
        buf_len += len(para)

    if buf:
        chunk_str = "\n\n".join(buf)
        chunks.append(MarkdownChunk(
            text=chunk_str,
            start_pos=curr_start,
            end_pos=curr_start + len(chunk_str),
        ))
    return chunks


def _split_oversized_section(section_text: str) -> list[str]:
    """將超過限制的單一段落以換行符二次切割。"""
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


def _bind_ancestor_paths(
    chunks: list[MarkdownChunk],
    skeleton_nodes: list[OutlineNode],
) -> None:
    """將每個切塊與全域大綱樹對齊，綁定活躍的祖先章節路徑與標題。"""
    if not skeleton_nodes:
        # 回退模式：若無大綱樹，使用相鄰 chunk 的 last_parent_heading
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.active_ancestor_path = chunks[i - 1].last_parent_heading
        return

    # 對於每個 chunk，尋找在 chunk.start_pos 之前最後出現的頂層或當前有效章節節點
    for chunk in chunks:
        active_node: OutlineNode | None = None
        for node in skeleton_nodes:
            if node.source_position <= chunk.start_pos:
                active_node = node
            else:
                break
        if active_node:
            chunk.active_ancestor_path = active_node.canonical_id
            chunk.active_ancestor_title = f"{active_node.canonical_id} {active_node.heading_title}".strip()
        elif skeleton_nodes:
            # 第一個 chunk 可能在第一個標題之前，預設賦予第 1 個節點
            chunk.active_ancestor_path = skeleton_nodes[0].canonical_id
            chunk.active_ancestor_title = f"{skeleton_nodes[0].canonical_id} {skeleton_nodes[0].heading_title}".strip()


# ═══════════════════════════════════════════════════════════════════════════
#  3. Pass 2: 帶祖先路徑的 LLM 萃取
# ═══════════════════════════════════════════════════════════════════════════

def _clean_llm_json(raw: str) -> str:
    """剝離 Markdown code fence 與首尾空白。"""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned.strip()


def _validate_extraction(data: object) -> list[dict]:
    """驗證 LLM 輸出的合法 JSON Array 格式。"""
    if not isinstance(data, list):
        raise ValueError(f"LLM 回傳非 JSON Array，實際類型：{type(data).__name__}")
    validated: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 筆資料非物件（dict），實際：{type(item).__name__}")
        if "question_id" not in item or "question_text" not in item:
            raise ValueError(
                f"第 {i} 筆資料缺少 question_id 或 question_text，"
                f"實際鍵值：{list(item.keys())}"
            )
        question_id = str(item["question_id"]).strip()
        question_text = str(item["question_text"]).strip()
        if question_id and question_text:
            validated.append({
                "question_id": question_id,
                "question_text": question_text,
            })
    return validated


def _deduplicate_items(items: list[dict]) -> list[dict]:
    """合併相同 question_id 的條目文本並保留原始順序。"""
    seen: dict[str, dict] = {}
    result: list[dict] = []
    for item in items:
        qid = item["question_id"]
        if qid in seen:
            existing = seen[qid]
            normalized_existing = re.sub(r"\s+", " ", existing["question_text"]).strip()
            normalized_new = re.sub(r"\s+", " ", item["question_text"]).strip()
            if normalized_new and normalized_new not in normalized_existing:
                existing["question_text"] = (
                    f"{existing['question_text']} {item['question_text']}".strip()
                )
        else:
            entry = {"question_id": qid, "question_text": item["question_text"]}
            seen[qid] = entry
            result.append(entry)
    return result


async def _extract_single_chunk(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    ancestor_path: str = "",
    ancestor_title: str = "",
) -> list[dict]:
    """Pass 2: 單區塊 LLM 萃取（注入全域祖先路徑與章節標題）。"""
    parts: list[str] = []
    if ancestor_path:
        parts.append(
            "<Active_Ancestor_Path>\n"
            f"當前文本屬於主章節：{ancestor_title or ancestor_path}\n"
            f"規範編號前綴為：{ancestor_path}\n"
            "【重要】此文本內所有子題目的 question_id 必須以此編號前綴展開（例如點分格式），切勿丟失主章節！\n"
            "</Active_Ancestor_Path>"
        )
    parts.append(f"<Markdown>\n{chunk_text}\n</Markdown>")
    user_prompt = "\n\n".join(parts)

    label = f"{chunk_index + 1}/{total_chunks}"
    logger.info(
        "v2 Pass 2 區塊萃取開始：%s  提示長度=%d 字元%s",
        label, len(user_prompt),
        f"（祖先路徑={ancestor_path}）" if ancestor_path else "",
    )
    start = time.perf_counter()

    raw_output = await chat_completion(
        prompt=user_prompt,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=4096,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("v2 Pass 2 區塊萃取完成：%s  耗時=%.1f ms", label, elapsed_ms)

    cleaned = _clean_llm_json(raw_output)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("v2 區塊 %s JSON 解析失敗：%s\n內容：%s", label, e, cleaned[:500])
        raise ValueError(f"區塊 {label} JSON 解析失敗：{e}") from e

    return _validate_extraction(parsed)


async def _extract_single_chunk_with_retry(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    ancestor_path: str = "",
    ancestor_title: str = "",
) -> list[dict]:
    """帶重試機制的區塊萃取。"""
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await _extract_single_chunk(
                chunk_text, chunk_index, total_chunks, ancestor_path, ancestor_title,
            )
        except ValueError as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "v2 區塊 %d/%d 萃取失敗（第 %d 次），重試中：%s",
                    chunk_index + 1, total_chunks, attempt + 1, e,
                )
    raise last_error  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
#  4. Pass 3: 全域一致性審查與驗證 (Consistency Verification)
# ═══════════════════════════════════════════════════════════════════════════

async def _verify_consistency(items: list[dict]) -> list[dict]:
    """Pass 3: 全域一致性檢核與修正。"""
    if len(items) > _CONSISTENCY_MAX_ITEMS:
        logger.info(
            "v2 題目數量 %d 超過一致性驗證上限 %d，跳過 LLM 驗證",
            len(items), _CONSISTENCY_MAX_ITEMS,
        )
        return items

    prompt = json.dumps(items, ensure_ascii=False, indent=2)
    logger.info("v2 Pass 3: 全域一致性驗證開始，題目=%d 筆", len(items))
    start = time.perf_counter()

    raw_output = await chat_completion(
        prompt=prompt,
        system_prompt=CONSISTENCY_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=8192,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("v2 Pass 3: 全域一致性驗證完成，耗時=%.1f ms", elapsed_ms)

    cleaned = _clean_llm_json(raw_output)
    try:
        verified = json.loads(cleaned)
        verified_items = _validate_extraction(verified)
        logger.info(
            "v2 一致性驗證結果：修正前=%d 筆  修正後=%d 筆",
            len(items), len(verified_items),
        )
        return verified_items
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("v2 一致性驗證失敗，使用原始結果：%s", e)
        return items


# ═══════════════════════════════════════════════════════════════════════════
#  5. 主公開介面 (extract_questions)
# ═══════════════════════════════════════════════════════════════════════════

async def extract_questions(markdown_text: str) -> list[dict]:
    """
    LLM-First 問卷萃取管線（方案 A：全域骨架驅動 + 祖先路徑並行萃取）。
    """
    pipeline_start = time.perf_counter()

    # ── 前處理 ──────────────────────────────────────────────────────────
    markdown_text = strip_table_of_contents(markdown_text)
    settings = get_settings()
    if settings.breakdown_nfkc_normalize:
        markdown_text = normalize_markdown_for_breakdown(
            markdown_text,
            linearize_tables=False,
        )

    # ── Step 1: 語義切塊 ──────────────────────────────────────────────
    chunks = split_markdown_semantically(markdown_text)
    logger.info(
        "v2 語義切塊完成：Markdown=%d 字元 → %d 個區塊（上限=%d 字元/塊）",
        len(markdown_text), len(chunks), _BREAKDOWN_CHUNK_SIZE,
    )

    # ── Step 2: Pass 1 全域章節骨架掃描（僅在多區塊時執行以省時）──────
    skeleton_nodes: list[OutlineNode] = []
    if len(chunks) > 1:
        skeleton_nodes = await _extract_document_skeleton(markdown_text)

    # 綁定祖先章節路徑至各區塊
    _bind_ancestor_paths(chunks, skeleton_nodes)

    # ── Step 3: Pass 2 並行 LLM 萃取 ──────────────────────────────────
    max_concurrency = settings.breakdown_max_concurrency
    sem = asyncio.Semaphore(max_concurrency)
    total = len(chunks)
    logger.info("v2 Breakdown LLM 並發上限：%d", max_concurrency)

    async def _guarded_extract(idx: int) -> list[dict]:
        async with sem:
            return await _extract_single_chunk_with_retry(
                chunks[idx].text,
                idx,
                total,
                chunks[idx].active_ancestor_path,
                chunks[idx].active_ancestor_title,
            )

    tasks = [_guarded_extract(i) for i in range(total)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Step 4: Reduce 合併 ───────────────────────────────────────────
    merged: list[dict] = []
    success_count = 0
    fail_count = 0

    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            fail_count += 1
            logger.warning(
                "v2 Reduce：區塊 %d/%d 萃取失敗（已跳過）：%s",
                i + 1, total, result,
            )
        else:
            success_count += 1
            merged.extend(result)

    if success_count == 0:
        raise ValueError(
            f"v2 全部 {total} 個區塊均萃取失敗，無法產出任何結果"
        )

    # ── Step 5: 去重合併 ─────────────────────────────────────────────
    merged = _deduplicate_items(merged)
    logger.info("v2 去重合併完成：%d 筆唯一題目", len(merged))

    # ── Step 6: Pass 3 一致性驗證 ────────────────────────────────────
    merged = await _verify_consistency(merged)

    elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info(
        "v2 Map-Reduce 完成：%d 區塊  成功=%d  失敗=%d  最終題目=%d 筆  耗時=%.1f ms",
        total, success_count, fail_count, len(merged), elapsed_ms,
    )

    return merged
