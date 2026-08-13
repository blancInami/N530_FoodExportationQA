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
from dataclasses import dataclass, field

from markitdown import MarkItDown

from app.config import get_settings
from app.services.llm import chat_completion
from app.services.breakdown_preprocessing import normalize_markdown_for_breakdown

logger = logging.getLogger(__name__)

# ── 切塊與併發參數 ────────────────────────────────────────────────────────
# 單一區塊送入 LLM 的最大字元數。留足餘量給 system prompt（~600 字）
# + 狀態注入（~100 字）+ 模型回應 token（max_tokens=4096）。
_BREAKDOWN_CHUNK_SIZE = 6_000

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
- 若提供 <Source_Candidates>，每一筆必須使用其中一個 "source_key"，並輸出 "question_text"。
    source_key 對應的 question_id 已由來源 Markdown 決定，不得自行重建、修改或猜測題號。
    同時輸出 role，值只能是 question、continuation 或 exclude。parent_requirement 一律為 continuation，
    並且必須使用與其 parent_heading 相同的 question_id。Candidate 的 kind 是來源分類，不能作為 role 值。
- 若未提供 <Source_Candidates>，輸出合法的 JSON Array of Objects，包含 "question_id" 與 "question_text" 兩個鍵值。
- 絕對禁止在 JSON 前後加上任何解釋性文字或 Markdown 標籤（如 ```json ）。\
"""

BREAKDOWN_VERIFICATION_SYSTEM_PROMPT = """\
你是資料萃取結果的驗證器。只能比對提供的 source_key，不得建立、修改或推測題號。
輸出單一 JSON Object，鍵為 confirmed_keys、missing_keys、duplicate_keys，值皆為字串陣列。
不得加入任何解釋或 Markdown 標籤。\
"""

_CHAPTER_HEADING_RE = re.compile(r"^\*\*(\d+)\.\s+(.+?)\*\*$")
_SUBHEADING_RE = re.compile(r"^\*\*(\d+(?:\.\d+)+)\s+(.+?)\*\*$")
_PARENTHESIZED_ITEM_RE = re.compile(r"^\*\*(\d+)\)\s+(.+?)\*\*$")
_INTEGER_CELL_RE = re.compile(r"^\d+$")
_DETAIL_CELL_RE = re.compile(r"^(\d+)\.(\d+)$")
_TOC_HEADING_RE = re.compile(r"^(?:目次|table of contents|contents)$", re.IGNORECASE)
_TOC_LINK_RE = re.compile(r"\]\(#(?:_Toc|toc)[^)]+\)", re.IGNORECASE)
_ROMAN_CHAPTER_RE = re.compile(r"^#{1,6}\s+.*?\b([IVXLCDM]+)[.．]\s", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(
    r"^(#{1,6})\s+(\d+(?:(?:[-－]|\.)\d+)*)(?:[.．])?(?=\s|$|[^\d])"
)
_LIST_NUMBER_RE = re.compile(r"(?<![\w.])(\d+)[.．](?=\s)")
_EXPLICIT_DECIMAL_ITEM_RE = re.compile(
    r"^(?:\|\s*)?(\d+(?:[.．]\d+)+)(?:[.．])?(?=\s|$)"
)
_PARENTHESIZED_PARENT_RE = re.compile(r"^(?:\|\s*)?[(（](\d+)[)）]")
_PARENTHESIZED_ROMAN_RE = re.compile(
    r"(?<!\w)[(（]([ivxlcdm]+)[)）]", re.IGNORECASE,
)
_ROMAN_SUBITEM_VALUES = frozenset({
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
})
_LETTER_ITEM_RE = re.compile(
    r"(?:^|\|)\s*([a-z])\.(?=\s|\||$|[^\x00-\x7f])", re.IGNORECASE,
)
_IMPLICIT_TABLE_LIST_RE = re.compile(r"^\|\s*\*\s*(\d+)[.．](?=\s)")


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


def _strip_table_of_contents_with_metadata(markdown_text: str) -> tuple[str, int, str]:
    """Remove an explicit Word-style table of contents without touching body links."""
    lines = markdown_text.splitlines(keepends=True)
    cleaned: list[str] = []
    in_table_of_contents = False
    removed_lines = 0

    for line in lines:
        plain_line = re.sub(r"[*#_`]+", "", line).strip()
        if not in_table_of_contents:
            if _TOC_HEADING_RE.fullmatch(plain_line):
                in_table_of_contents = True
                removed_lines += 1
                continue
            cleaned.append(line)
            continue

        if not plain_line or _TOC_LINK_RE.search(line):
            removed_lines += 1
            continue

        in_table_of_contents = False
        cleaned.append(line)

    strategy = "explicit_heading" if removed_lines else "none"
    return "".join(cleaned), removed_lines, strategy


def strip_table_of_contents(markdown_text: str) -> str:
    """Remove explicit MarkItDown Word table-of-contents content from Markdown."""
    cleaned, removed_lines, strategy = _strip_table_of_contents_with_metadata(markdown_text)
    if removed_lines:
        logger.info("已移除 Word 目錄：策略=%s  行數=%d", strategy, removed_lines)
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
#  2. 語義切塊（Semantic Chunking）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NumberingState:
    """The deterministic hierarchical numbering context at a document position."""
    chapter: str = ""
    numeric_path: tuple[str, ...] = ()
    numeric_levels: tuple[int, ...] = ()
    parenthesized: str = ""
    letter: str = ""
    roman_subitem: str = ""

    def parent_prefix(self) -> str:
        """Return the active parent path, excluding a completed letter item."""
        parts = [self.chapter, *self.numeric_path, self.parenthesized]
        return ".".join(part for part in parts if part)


@dataclass
class MarkdownChunk:
    """語義切塊結果。

    Attributes:
        text: 該區塊的 Markdown 文本。
        start_numbering_state: 該區塊開始前已知的完整題號狀態。
        end_numbering_state: 掃描該區塊後的完整題號狀態，供下一塊繼承。
    """
    text: str
    last_parent_heading: str = ""
    start_numbering_state: NumberingState = field(default_factory=NumberingState)
    end_numbering_state: NumberingState = field(default_factory=NumberingState)


@dataclass(frozen=True)
class NumberingSegment:
    """An LLM work unit with the explicit numbering context of its section."""
    text: str
    numbering_state: NumberingState
    scope_reset: bool = False


@dataclass(frozen=True)
class SourceQuestionCandidate:
    """A parser-owned question ID available for an LLM extraction segment."""
    source_key: str
    question_id: str
    english_text: str = ""
    kind: str = "question"
    source_line: int = 0


@dataclass(frozen=True)
class SourceKeyCoverageReport:
    """Deterministic source-key coverage diagnostics for one LLM segment."""
    selected_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    duplicate_keys: tuple[str, ...]
    verification: dict[str, object] | None = None

    @property
    def is_complete(self) -> bool:
        return not self.missing_keys and not self.duplicate_keys


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


def _scan_numbering_state(text: str, state: NumberingState) -> NumberingState:
    """Advance numbering state using only explicit document markers."""
    for raw_line in text.splitlines():
        plain_line = re.sub(r"[*_`]", "", raw_line).strip()
        roman_match = _ROMAN_CHAPTER_RE.match(raw_line)
        if roman_match:
            state = NumberingState(chapter=roman_match.group(1).upper())
            continue

        heading_match = _NUMBERED_HEADING_RE.match(plain_line)
        if heading_match:
            heading_level = len(heading_match.group(1))
            label_parts = tuple(re.split(r"[-－.]", heading_match.group(2)))
            ancestor_parts = tuple(
                part
                for part, level in zip(state.numeric_path, state.numeric_levels)
                if level < heading_level
            )
            ancestor_levels = tuple(
                level for level in state.numeric_levels if level < heading_level
            )
            state = NumberingState(
                chapter=state.chapter,
                numeric_path=(*ancestor_parts, *label_parts),
                numeric_levels=(*ancestor_levels, *(heading_level for _ in label_parts)),
            )
            continue

        decimal_item_match = _EXPLICIT_DECIMAL_ITEM_RE.match(plain_line)
        if decimal_item_match:
            label_parts = tuple(re.split(r"[.．]", decimal_item_match.group(1)))
            state = NumberingState(
                chapter=state.chapter,
                numeric_path=label_parts,
                numeric_levels=(999,) * len(label_parts),
            )
            continue

        for match in _LIST_NUMBER_RE.finditer(plain_line):
            number = match.group(1)
            if len(state.numeric_path) > 1:
                numeric_path = (*state.numeric_path[:-1], number)
                numeric_levels = (*state.numeric_levels[:-1], 999)
            else:
                numeric_path = (*state.numeric_path, number)
                numeric_levels = (*state.numeric_levels, 999)
            state = NumberingState(
                chapter=state.chapter,
                numeric_path=numeric_path,
                numeric_levels=numeric_levels,
            )

        parenthesized_match = _PARENTHESIZED_PARENT_RE.match(plain_line)
        if parenthesized_match:
            state = NumberingState(
                chapter=state.chapter,
                numeric_path=state.numeric_path,
                numeric_levels=state.numeric_levels,
                parenthesized=parenthesized_match.group(1),
            )

        letter_match = _LETTER_ITEM_RE.search(raw_line)
        if letter_match:
            state = NumberingState(
                chapter=state.chapter,
                numeric_path=state.numeric_path,
                numeric_levels=state.numeric_levels,
                parenthesized=state.parenthesized,
                letter=letter_match.group(1).lower(),
            )

        for match in _inline_roman_subitem_matches(raw_line, letter_match):
            state = NumberingState(
                chapter=state.chapter,
                numeric_path=state.numeric_path,
                numeric_levels=state.numeric_levels,
                parenthesized=state.parenthesized,
                letter=state.letter,
                roman_subitem=match.group(1).lower(),
            )

    return state


def _inline_roman_subitem_matches(
    raw_line: str, letter_match: re.Match[str] | None,
) -> list[re.Match[str]]:
    """Return list markers while excluding prose references such as ``c (ii)``."""
    if letter_match is None:
        return []

    matches: list[re.Match[str]] = []
    for match in _PARENTHESIZED_ROMAN_RE.finditer(raw_line):
        if match.group(1).lower() not in _ROMAN_SUBITEM_VALUES:
            continue
        preceding = raw_line[letter_match.end():match.start()].rstrip()
        if re.search(r"\b[a-z]\s*$", preceding, re.IGNORECASE):
            continue
        following = raw_line[match.end():].lstrip()
        if not following or following.startswith(("(", "（")):
            continue
        if following[0].isascii() and following[0].islower():
            continue
        if re.search(r"\([a-z]+\)\s*$", preceding, re.IGNORECASE) and (
            following[0].isascii() and following[0].islower()
        ):
            continue
        matches.append(match)
    return matches


def _attach_numbering_states(chunks: list[MarkdownChunk]) -> None:
    """Attach sequential start/end numbering states to precomputed chunks."""
    state = NumberingState()
    for chunk in chunks:
        chunk.start_numbering_state = state
        state = _scan_numbering_state(chunk.text, state)
        chunk.end_numbering_state = state


def _split_numbering_segments(chunks: list[MarkdownChunk]) -> list[NumberingSegment]:
    """Split chunks at explicit Markdown heading scope changes.

    A chunk can contain both ``I.`` and ``II.`` when it is smaller than the
    size limit. Its start state is then empty, even though all items below the
    first heading require the ``I`` prefix. Roman and numbered Markdown headings
    are explicit scope-reset points, so each section gets deterministic context
    without splitting ordinary table rows into individual LLM calls.
    """
    segments: list[NumberingSegment] = []
    for chunk in chunks:
        lines = chunk.text.splitlines(keepends=True)
        boundaries = []
        for index, line in enumerate(lines):
            plain_line = re.sub(r"[*_`]", "", line).strip()
            if _ROMAN_CHAPTER_RE.match(line) or _NUMBERED_HEADING_RE.match(plain_line):
                boundaries.append(index)
        if not boundaries:
            segments.append(NumberingSegment(chunk.text, chunk.start_numbering_state))
            continue

        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        boundaries.append(len(lines))
        state = chunk.start_numbering_state

        for start, end in zip(boundaries, boundaries[1:]):
            text = "".join(lines[start:end]).strip()
            if not text:
                continue
            first_line = lines[start] if start < len(lines) else ""
            entry_state = _scan_numbering_state(first_line, state)
            plain_first_line = re.sub(r"[*_`]", "", first_line).strip()
            scope_reset = bool(_NUMBERED_HEADING_RE.match(plain_first_line))
            segments.append(NumberingSegment(text, entry_state, scope_reset))
            state = _scan_numbering_state(text, state)

    return segments


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
    chunks = chunks or [MarkdownChunk(text=markdown, last_parent_heading="")]
    _attach_numbering_states(chunks)
    return chunks


def _split_plain_text(text: str) -> list[MarkdownChunk]:
    """無 Markdown 標題時的回退切割：以段落（連續空行）分界。"""
    if len(text) <= _BREAKDOWN_CHUNK_SIZE:
        chunks = [MarkdownChunk(text=text, last_parent_heading="")]
        _attach_numbering_states(chunks)
        return chunks

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
    _attach_numbering_states(chunks)
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
#  3. Deterministic questionnaire-table parsing
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_markdown_cell(cell: str) -> str:
    """Remove table and emphasis markup while preserving the source text."""
    text = re.sub(r"\*{1,3}", "", cell)
    return re.sub(r"\s+", " ", text).strip()


def _table_cells(line: str) -> list[str]:
    """Return non-empty Markdown table cells in their original order."""
    if not line.lstrip().startswith("|"):
        return []
    cells = [_normalize_markdown_cell(cell) for cell in line.strip().split("|")[1:-1]]
    return [cell for cell in cells if cell]


def _text_after_number(cells: list[str], start: int) -> str:
    """Collect a table item's text until the next numbered cell."""
    text_cells: list[str] = []
    for cell in cells[start + 1:]:
        if _INTEGER_CELL_RE.fullmatch(cell) or _DETAIL_CELL_RE.fullmatch(cell):
            break
        text_cells.append(cell)
    return " ".join(text_cells).strip()


def extract_structured_questions(markdown_text: str) -> list[dict]:
    """Extract explicitly numbered questionnaire items from MarkItDown tables.

    Word questionnaires can place a chapter number in a heading and question
    numbers in separate table cells. Keeping those source numbers deterministic
    avoids asking the LLM to reconstruct hierarchical IDs.
    """
    chapter: str | None = None
    last_main_number: str | None = None
    last_subheading: str | None = None
    items: list[dict] = []
    seen_ids: set[str] = set()
    found_question_table = False

    def add_item(question_id: str, question_text: str) -> None:
        if not question_text or question_id in seen_ids:
            return
        seen_ids.add(question_id)
        items.append({"question_id": question_id, "question_text": question_text})

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        chapter_match = _CHAPTER_HEADING_RE.fullmatch(line)
        if chapter_match:
            chapter = chapter_match.group(1)
            last_main_number = None
            last_subheading = None
            continue

        subheading_match = _SUBHEADING_RE.fullmatch(line)
        if subheading_match:
            last_subheading = subheading_match.group(1)
            continue

        parenthesized_match = _PARENTHESIZED_ITEM_RE.fullmatch(line)
        if parenthesized_match and last_subheading:
            add_item(
                f"{last_subheading}.{parenthesized_match.group(1)}",
                parenthesized_match.group(2),
            )
            continue

        cells = _table_cells(raw_line)
        if not cells or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue

        if any("detailed questions" in cell.lower() for cell in cells):
            found_question_table = True
            continue

        if chapter is None or not found_question_table:
            continue

        for index, cell in enumerate(cells):
            if _INTEGER_CELL_RE.fullmatch(cell):
                question_text = _text_after_number(cells, index)
                if question_text:
                    last_main_number = cell
                    add_item(f"{chapter}.{cell}", question_text)
                continue

            detail_match = _DETAIL_CELL_RE.fullmatch(cell)
            if detail_match and last_main_number:
                question_text = _text_after_number(cells, index)
                if question_text:
                    add_item(
                        f"{chapter}.{last_main_number}.{detail_match.group(1)}.{detail_match.group(2)}",
                        question_text,
                    )

    if found_question_table and len(items) >= 3:
        logger.info("結構化表格解析完成：萃取題目=%d 筆", len(items))
        return items
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  4. LLM 萃取 + JSON 清洗驗證
# ═══════════════════════════════════════════════════════════════════════════

def _clean_llm_json(raw: str) -> str:
    """剝離 Markdown code fence 與首尾空白。"""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned.strip()


def _validate_breakdown(data: object) -> list[dict]:
    """Validate LLM items using source_key or the legacy question_id field."""
    if not isinstance(data, list):
        raise ValueError(f"LLM 回傳非 JSON Array，實際類型：{type(data).__name__}")
    validated: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 筆資料非物件（dict），實際：{type(item).__name__}")
        if "question_text" not in item or (
            "source_key" not in item and "question_id" not in item
        ):
            raise ValueError(
                f"第 {i} 筆資料缺少必要欄位 source_key 或 question_id / question_text，"
                f"實際鍵值：{list(item.keys())}"
            )
        role = str(item.get("role", "question")).strip().lower()
        role = {
            "parent_requirement": "continuation",
            "parent_heading": "question",
            "letter_item": "question",
            "inline_subitem": "question",
            "explicit_item": "question",
            "implicit_list_item": "question",
        }.get(role, role)
        if role not in {"question", "continuation", "exclude"}:
            raise ValueError(f"第 {i} 筆資料 role 無效：{role}")
        validated.append({
            "question_id": str(item.get("question_id", "")).strip(),
            "question_text": str(item["question_text"]).strip(),
            "source_key": str(item.get("source_key", "")).strip(),
            "role": role,
        })
    return validated


def _merge_items_by_question_id(items: list[dict]) -> list[dict]:
    """Merge only explicit continuations into their preceding source question."""
    merged: list[dict] = []
    for item in items:
        if item["role"] == "exclude":
            continue
        if item["role"] != "continuation":
            merged.append({
                "question_id": item["question_id"],
                "question_text": item["question_text"],
            })
            continue

        target = next(
            (
                existing for existing in reversed(merged)
                if existing["question_id"] == item["question_id"]
            ),
            None,
        )
        if target is None:
            merged.append({
                "question_id": item["question_id"],
                "question_text": item["question_text"],
            })
            continue
        normalized_existing = re.sub(r"\s+", " ", target["question_text"]).strip()
        normalized_new = re.sub(r"\s+", " ", item["question_text"]).strip()
        if normalized_new and normalized_new not in normalized_existing:
            target["question_text"] = f"{target['question_text']} {item['question_text']}".strip()
    return merged


def _has_english_content(text: str) -> bool:
    """Require enough Latin text to avoid creating candidates from Japanese-only lines."""
    return sum(char.isascii() and char.isalpha() for char in text) >= 3


def _build_source_candidates(
    source_text: str, entry_state: NumberingState,
) -> list[SourceQuestionCandidate]:
    """Build parser-owned IDs plus Markdown semantic context for one LLM segment."""
    candidates: list[SourceQuestionCandidate] = []
    inline_question_ids: set[str] = set()
    implicit_list_marker = ""
    implicit_list_base = ""
    implicit_list_next = 0
    state = entry_state

    def add_candidate(
        question_id: str, text: str, kind: str, source_line: int,
    ) -> None:
        if not question_id or not _has_english_content(text):
            return
        candidate = SourceQuestionCandidate(
            f"C{len(candidates) + 1:03d}", question_id, text.strip(), kind, source_line,
        )
        if candidate not in candidates:
            candidates.append(candidate)

    for source_line, raw_line in enumerate(source_text.splitlines(), start=1):
        implicit_list_match = _IMPLICIT_TABLE_LIST_RE.match(raw_line)
        state_before_line = state
        state = _scan_numbering_state(raw_line, state)
        parent_prefix = state.parent_prefix()
        if implicit_list_match:
            marker = implicit_list_match.group(1)
            if marker != implicit_list_marker:
                implicit_list_base = state_before_line.parent_prefix()
                implicit_list_next = int(marker)
                implicit_list_marker = marker
            if implicit_list_base:
                add_candidate(
                    f"{implicit_list_base}.{implicit_list_next}",
                    raw_line,
                    "implicit_list_item",
                    source_line,
                )
                implicit_list_next += 1
                continue
        else:
            implicit_list_marker = ""
            implicit_list_base = ""
            implicit_list_next = 0
        if re.match(r"^#{1,6}\s", raw_line):
            continue
        if _EXPLICIT_DECIMAL_ITEM_RE.match(raw_line.strip()):
            add_candidate(parent_prefix, raw_line, "explicit_item", source_line)
            continue
        if _PARENTHESIZED_PARENT_RE.match(raw_line.strip()):
            add_candidate(parent_prefix, raw_line, "parent_heading", source_line)
            continue

        letter_match = _LETTER_ITEM_RE.search(raw_line)
        if letter_match and parent_prefix:
            letter_id = f"{parent_prefix}.{letter_match.group(1).lower()}"
            add_candidate(letter_id, raw_line, "letter_item", source_line)
            for roman_match in _inline_roman_subitem_matches(raw_line, letter_match):
                inline_question_id = f"{letter_id}.{roman_match.group(1).lower()}"
                if inline_question_id not in inline_question_ids:
                    inline_question_ids.add(inline_question_id)
                    add_candidate(
                        inline_question_id, raw_line[roman_match.start():],
                        "inline_subitem", source_line,
                    )
            continue

        if parent_prefix and raw_line.strip() and _has_english_content(raw_line):
            add_candidate(parent_prefix, raw_line, "parent_requirement", source_line)

    return candidates


async def _verify_source_key_coverage(
    source_candidates: list[SourceQuestionCandidate], selected_keys: list[str],
    chunk_index: int, total_chunks: int,
) -> SourceKeyCoverageReport:
    """Ask the LLM to review anomalous key coverage without changing output IDs."""
    selected_key_set = set(selected_keys)
    duplicate_keys = sorted(
        key for key in selected_key_set if selected_keys.count(key) > 1
    )
    missing_keys = sorted(
        candidate.source_key
        for candidate in source_candidates
        if candidate.source_key not in selected_key_set
    )
    report = SourceKeyCoverageReport(
        selected_keys=tuple(selected_keys),
        missing_keys=tuple(missing_keys),
        duplicate_keys=tuple(duplicate_keys),
    )
    if not get_settings().breakdown_llm_verify or not selected_keys:
        return report
    if not duplicate_keys and not missing_keys:
        return report

    candidate_lines = "\n".join(
        f'{candidate.source_key}: {candidate.question_id}'
        for candidate in source_candidates
    )
    prompt = (
        "<Source_Candidates>\n"
        f"{candidate_lines}\n"
        "</Source_Candidates>\n"
        "<Extraction_Result>\n"
        f"selected_keys={json.dumps(selected_keys)}\n"
        f"detected_missing_keys={json.dumps(missing_keys)}\n"
        f"detected_duplicate_keys={json.dumps(duplicate_keys)}\n"
        "</Extraction_Result>"
    )
    raw_output = await chat_completion(
        prompt=prompt,
        system_prompt=BREAKDOWN_VERIFICATION_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=512,
    )
    try:
        verification = json.loads(_clean_llm_json(raw_output))
        if not isinstance(verification, dict):
            raise ValueError("驗證回傳非 JSON Object")
        reported_keys = {
            key
            for field in ("confirmed_keys", "missing_keys", "duplicate_keys")
            for key in verification.get(field, [])
            if isinstance(key, str)
        }
        valid_keys = {candidate.source_key for candidate in source_candidates}
        if not reported_keys <= valid_keys:
            raise ValueError("驗證回傳未知的 source_key")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("區塊 %d/%d 二次驗證結果無效：%s", chunk_index + 1, total_chunks, exc)
        return report

    logger.warning(
        "區塊 %d/%d 題號候選需人工檢視：遺漏=%s 重複=%s LLM驗證=%s",
        chunk_index + 1, total_chunks, missing_keys, duplicate_keys, verification,
    )
    return SourceKeyCoverageReport(
        selected_keys=report.selected_keys,
        missing_keys=report.missing_keys,
        duplicate_keys=report.duplicate_keys,
        verification=verification,
    )


async def _extract_single_chunk(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    state_prefix: str | None = None,
    numbering_state: NumberingState | None = None,
    scope_reset: bool = False,
    source_candidates: list[SourceQuestionCandidate] | None = None,
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
    if source_candidates:
        candidates_xml = "\n".join(
            f'<Candidate source_key="{candidate.source_key}" question_id="{candidate.question_id}" '
            f'kind="{candidate.kind}" source_line="{candidate.source_line}">'
            f'{candidate.english_text}'
            "</Candidate>"
            for candidate in source_candidates
        )
        parts.append(
            "<Source_Candidates>\n"
            "以下題號由來源結構固定。僅可回傳列出的 source_key；"
            "不要輸出或推測 question_id。\n"
            f"{candidates_xml}\n"
            "</Source_Candidates>"
        )
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

    items = _validate_breakdown(parsed)
    candidate_by_key = {
        candidate.source_key: candidate
        for candidate in source_candidates or []
    }
    selected_keys = [item["source_key"] for item in items if item["source_key"]]
    for item in items:
        if item["source_key"]:
            candidate = candidate_by_key.get(item["source_key"])
            if candidate is None:
                raise ValueError(f"區塊 {label} 回傳未知 source_key：{item['source_key']}")
            item["question_id"] = candidate.question_id
            if candidate.kind == "parent_requirement":
                item["role"] = "continuation"
        item.pop("source_key", None)
    if source_candidates:
        await _verify_source_key_coverage(
            source_candidates, selected_keys, chunk_index, total_chunks,
        )
    if numbering_state:
        _restore_known_numbering_prefix(
            items, numbering_state, scope_reset, chunk_text,
        )
    return _merge_items_by_question_id(items)


def _restore_known_numbering_prefix(
    items: list[dict],
    state: NumberingState,
    scope_reset: bool = False,
    source_text: str = "",
) -> None:
    """Restore a prefix only where the chunk's explicit state proves it."""
    parent_prefix = state.parent_prefix()
    prefix_parts = parent_prefix.split(".") if parent_prefix else []
    if not prefix_parts:
        return

    for item in items:
        source_question_id = _source_question_id(
            source_text, item["question_text"], state,
        )
        if source_question_id:
            item["question_id"] = source_question_id
            continue
        question_parts = [part for part in item["question_id"].split(".") if part]
        if not question_parts:
            continue
        if state.chapter and not state.numeric_path and not state.parenthesized:
            if question_parts[0] != state.chapter:
                item["question_id"] = ".".join([state.chapter, *question_parts])
            continue
        if scope_reset and question_parts[0] == state.chapter:
            state_path = list(state.numeric_path)
            question_path = question_parts[1:]
            for index in range(len(question_path) - len(state_path) + 1):
                if question_path[index:index + len(state_path)] == state_path:
                    item["question_id"] = ".".join([
                        state.chapter,
                        *state_path,
                        *question_path[index + len(state_path):],
                    ])
                    break
        overlap = min(len(prefix_parts), len(question_parts))
        while overlap and prefix_parts[-overlap:] != question_parts[:overlap]:
            overlap -= 1
        if overlap:
            item["question_id"] = ".".join([*prefix_parts, *question_parts[overlap:]])
        elif len(question_parts) == 1 and question_parts[0].isalpha():
            item["question_id"] = ".".join([*prefix_parts, question_parts[0]])


def _source_question_id(
    source_text: str, question_text: str, entry_state: NumberingState,
) -> str | None:
    """Return the explicit source ID when an LLM item's text identifies its line."""
    normalized_question = re.sub(r"[^a-z0-9]+", "", question_text.lower())
    if len(normalized_question) < 24:
        return None

    state = entry_state
    for raw_line in source_text.splitlines():
        state = _scan_numbering_state(raw_line, state)
        normalized_line = re.sub(r"[^a-z0-9]+", "", raw_line.lower())
        if normalized_question not in normalized_line:
            continue
        parent_prefix = state.parent_prefix()
        if not parent_prefix:
            continue
        if not state.letter:
            if _EXPLICIT_DECIMAL_ITEM_RE.match(raw_line.strip()):
                return parent_prefix
            if _PARENTHESIZED_PARENT_RE.match(raw_line.strip()):
                return parent_prefix
            continue

        letter_id = f"{parent_prefix}.{state.letter}"
        letter_match = _LETTER_ITEM_RE.search(raw_line)
        roman_matches = _inline_roman_subitem_matches(raw_line, letter_match)
        for index, match in enumerate(roman_matches):
            end = roman_matches[index + 1].start() if index + 1 < len(roman_matches) else len(raw_line)
            normalized_subitem = re.sub(r"[^a-z0-9]+", "", raw_line[match.start():end].lower())
            if normalized_question in normalized_subitem:
                return f"{letter_id}.{match.group(1).lower()}"
        return letter_id
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  5. Map-Reduce 萃取管線（公開介面）
# ═══════════════════════════════════════════════════════════════════════════

async def extract_questions(markdown_text: str) -> list[dict]:
    """
    從 Markdown 文本萃取結構化題號與英文題目。

    管線流程（Map-Reduce with State Injection）：

    **Map 階段**
      1. ``split_markdown_semantically()`` 將 Markdown 依標題層級拆為
         多個 ``MarkdownChunk``，每塊 ≤ ``_BREAKDOWN_CHUNK_SIZE`` 字元。
        2. 為每個區塊建立 LLM 推論任務，動態注入該區塊開始前已掃描的
            完整題號狀態，讓非標題的章節編號跨切塊仍可被追溯。
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
    markdown_text = strip_table_of_contents(markdown_text)
    settings = get_settings()
    if settings.breakdown_nfkc_normalize:
        markdown_text = normalize_markdown_for_breakdown(
            markdown_text,
            linearize_tables=False,
        )
    if settings.breakdown_table_linearization_mode != "off":
        logger.warning(
            "表格線性化尚未接入主解析器，已忽略設定：mode=%s",
            settings.breakdown_table_linearization_mode,
        )

    structured_items = extract_structured_questions(markdown_text)
    if structured_items:
        logger.info(
            "使用結構化表格快速路徑：略過 LLM 萃取，題目=%d 筆",
            len(structured_items),
        )
        return structured_items

    # ── Step 1: 語義切塊 ──────────────────────────────────────────────────
    chunks = split_markdown_semantically(markdown_text)
    logger.info(
        "語義切塊完成：Markdown=%d 字元 → %d 個區塊（上限=%d 字元/塊）",
        len(markdown_text), len(chunks), _BREAKDOWN_CHUNK_SIZE,
    )

    # ── Step 2 & 3: Map — 建立並行任務（含跨區塊狀態注入）────────────────
    segments = _split_numbering_segments(chunks)
    max_concurrency = settings.breakdown_max_concurrency
    sem = asyncio.Semaphore(max_concurrency)
    total = len(segments)
    logger.info("Breakdown LLM 並發上限：%d", max_concurrency)

    async def _guarded_extract(idx: int) -> list[dict]:
        """受 Semaphore 控制的單區塊萃取。"""
        # 以文件順序掃描出的題號狀態，保留不會出現在 Markdown heading 的層級。
        state_prefix: str | None = None
        parent_prefix = segments[idx].numbering_state.parent_prefix()
        if parent_prefix:
            state_prefix = (
                "<Numbering_Context>\n"
                f"此文本接續前文；已由來源明確辨識的父題號為 {parent_prefix}。\n"
                "子項目必須接續此父題號；不得捨棄任何既有層級。\n"
                "</Numbering_Context>"
            )
        async with sem:
            return await _extract_single_chunk(
                segments[idx].text, idx, total, state_prefix,
                segments[idx].numbering_state,
                segments[idx].scope_reset,
                _build_source_candidates(
                    segments[idx].text, segments[idx].numbering_state,
                ),
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
