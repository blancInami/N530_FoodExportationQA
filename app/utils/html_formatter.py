"""
html_formatter.py — 專用於問卷解析題目的 HTML 複合格式化與容錯工具庫。

提供：
  - convert_markdown_table_to_html(text)  : 將 Markdown Pipe Table 轉為語意化 <table>
  - convert_markdown_checklist_to_html(text): 將 Markdown 勾選清單轉為 <ul><li>
  - merge_html_composite_texts(text1, text2): 跨頁 HTML 表格與清單智慧縫合
  - format_composite_question_text(raw_text) : 全域容錯後處理器
  - sanitize_and_balance_html(html_text)   : 標籤平衡與安全清洗
"""
import html
import re


# ── Markdown Pipe 表格匹配與轉換 ──────────────────────────────────────────────

_PIPE_TABLE_PATTERN = re.compile(
    r"(?:^[ \t]*\|[^\n]+\|[ \t]*\n)+"  # 連續多行以 | 開頭和結尾的表格列
    r"[ \t]*\|[^\n]+\|[ \t]*",
    re.MULTILINE
)


def convert_markdown_table_to_html(text: str) -> str:
    """
    掃描文字中的 Markdown Pipe 表格區塊，將其轉換為標準語意 HTML <table>。
    保留表格前置與後置的問句文字。
    """
    if not text or "|" not in text:
        return text

    def _replace_table(match: re.Match) -> str:
        table_raw = match.group(0).strip()
        lines = [line.strip() for line in table_raw.splitlines() if line.strip()]
        if len(lines) < 2:
            return table_raw

        # 解析每一列的單元格
        rows: list[list[str]] = []
        for line in lines:
            if line.startswith("|"):
                line = line[1:]
            if line.endswith("|"):
                line = line[:-1]
            cells = [c.strip() for c in line.split("|")]
            rows.append(cells)

        # 尋找對齊分隔列（如 | --- | :---: | ---: |）
        header_idx = -1
        for i, row in enumerate(rows):
            if all(re.match(r"^:?-+:?$", cell) for cell in row if cell):
                header_idx = i
                break

        html_out = ['<table border="1">']
        if header_idx > 0:
            # 有分隔線：分隔線上方為表頭 thead
            headers = rows[:header_idx]
            body_rows = rows[header_idx + 1:]

            html_out.append("  <thead>")
            for hrow in headers:
                html_out.append("    <tr>")
                for cell in hrow:
                    html_out.append(f"      <th>{html.escape(cell)}</th>")
                html_out.append("    </tr>")
            html_out.append("  </thead>")
        else:
            # 無標準分隔線：第一列預設為 header，其餘為 body
            body_rows = rows

        if body_rows:
            html_out.append("  <tbody>")
            for brow in body_rows:
                html_out.append("    <tr>")
                for cell in brow:
                    html_out.append(f"      <td>{html.escape(cell)}</td>")
                html_out.append("    </tr>")
            html_out.append("  </tbody>")

        html_out.append("</table>")
        return "\n" + "\n".join(html_out) + "\n"

    return _PIPE_TABLE_PATTERN.sub(_replace_table, text).strip()


# ── Markdown 勾選清單 / 選項列表轉換 ──────────────────────────────────────────

_CHECKLIST_BLOCK_PATTERN = re.compile(
    r"(?:^[ \t]*(?:[-*•]|\d+\.)[ \t]+(?:\[[ xX ]?\][ \t]*)?[^\n]+\n?)+",
    re.MULTILINE
)


def convert_markdown_checklist_to_html(text: str) -> str:
    """
    將文字中的 Markdown 勾選清單或條列選項轉換為語意化 <ul><li> 或 <ol><li>。
    保留問句本體與 HTML 結構。
    """
    if not text:
        return text

    # 若已經包含 <ul> 或 <ol>，避免重複包裝
    if "<ul" in text.lower() or "<ol" in text.lower():
        return text

    def _replace_list(match: re.Match) -> str:
        block = match.group(0).strip()
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            return block

        is_ordered = bool(re.match(r"^\d+\.", lines[0]))
        tag = "ol" if is_ordered else "ul"

        items = []
        for line in lines:
            # 去除前綴符號 (- / * / • / 1.)
            cleaned_line = re.sub(r"^(?:[-*•]|\d+\.)\s+", "", line).strip()
            if cleaned_line:
                items.append(f"  <li>{cleaned_line}</li>")

        if not items:
            return block

        return f"\n<{tag}>\n" + "\n".join(items) + f"\n</{tag}>\n"

    # 僅在確實有條列格式且有前置或後置引導詞時處理
    return _CHECKLIST_BLOCK_PATTERN.sub(_replace_list, text).strip()


# ── HTML 標籤平衡與安全清洗 ───────────────────────────────────────────────────

def sanitize_and_balance_html(html_text: str) -> str:
    """
    修復未正確閉合的常見 HTML 標籤（table, thead, tbody, tr, th, td, ul, ol, li, p）。
    過濾危險標籤與無效嵌套。
    """
    if not html_text:
        return ""

    text = html_text.strip()

    # 移除危險標籤
    text = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*style[^>]*>.*?<\s*/\s*style\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # 標籤自動閉合檢查
    pair_tags = ["table", "thead", "tbody", "tr", "th", "td", "ul", "ol", "li", "p"]
    for tag in pair_tags:
        open_count = len(re.findall(rf"<\s*{tag}(?:\s+[^>]*)?>", text, re.IGNORECASE))
        close_count = len(re.findall(rf"<\s*/\s*{tag}\s*>", text, re.IGNORECASE))
        if open_count > close_count:
            # 補齊遺失的閉合標籤
            text += f"</{tag}>" * (open_count - close_count)

    return text.strip()


# ── 跨頁 HTML 表格與清單智慧縫合 ──────────────────────────────────────────────

def merge_html_composite_texts(text1: str, text2: str) -> str:
    """
    跨頁 HTML 複合題目智慧縫合：
    1. 若兩者皆包含 <table>，將 text2 的 <tbody> 內部列追加至 text1 的 <tbody> 中，產出單一完整 <table>。
    2. 若兩者皆包含 <ul> 或 <ol>，合併其 <li> 項目。
    3. 其餘情況以段落/空格乾淨連接，並避免重複文字。
    """
    if not text1:
        return text2 or ""
    if not text2:
        return text1 or ""

    t1 = text1.strip()
    t2 = text2.strip()

    # 情況 1：兩者皆包含表格 -> 合併 tbody 列
    if "<table" in t1.lower() and "<table" in t2.lower():
        # 提取 t2 中的 <tr> 列
        t2_rows = re.findall(r"<\s*tr[^>]*>.*?<\s*/\s*tr\s*>", t2, flags=re.IGNORECASE | re.DOTALL)
        if t2_rows:
            # 檢查 t2 的第一列是否為重複表頭
            t1_first_row = re.search(r"<\s*tr[^>]*>.*?<\s*/\s*tr\s*>", t1, flags=re.IGNORECASE | re.DOTALL)
            if t1_first_row and t2_rows:
                t1_header_str = re.sub(r"\s+", " ", t1_first_row.group(0)).strip().lower()
                t2_header_str = re.sub(r"\s+", " ", t2_rows[0]).strip().lower()
                if t1_header_str == t2_header_str:
                    # 重複表頭，捨棄 t2 的第一列
                    t2_rows = t2_rows[1:]

            if t2_rows:
                new_rows_str = "\n    " + "\n    ".join(t2_rows)
                # 尋找 t1 的 </tbody> 結尾或 </table> 結尾進行插入
                if "</tbody>" in t1:
                    t1 = t1.replace("</tbody>", f"{new_rows_str}\n  </tbody>", 1)
                elif "</table>" in t1:
                    t1 = t1.replace("</table>", f"  <tbody>{new_rows_str}\n  </tbody>\n</table>", 1)
                else:
                    t1 = t1 + f"\n{new_rows_str}\n</table>"

                # 提取 t2 表格外的非表格文字（若有）
                t2_non_table = re.sub(r"<\s*table[^>]*>.*?<\s*/\s*table\s*>", "", t2, flags=re.IGNORECASE | re.DOTALL).strip()
                if t2_non_table and t2_non_table not in t1:
                    t1 = f"{t1}\n<p>{t2_non_table}</p>"

                return sanitize_and_balance_html(t1)

    # 情況 2：兩者皆包含 <ul> 或 <ol> -> 合併 <li> 項目
    for list_tag in ["ul", "ol"]:
        if f"<{list_tag}" in t1.lower() and f"<{list_tag}" in t2.lower():
            t2_items = re.findall(r"<\s*li[^>]*>.*?<\s*/\s*li\s*>", t2, flags=re.IGNORECASE | re.DOTALL)
            if t2_items:
                new_items_str = "\n  " + "\n  ".join(t2_items)
                t1 = t1.replace(f"</{list_tag}>", f"{new_items_str}\n</{list_tag}>", 1)
                return sanitize_and_balance_html(t1)

    # 情況 3：一般文字與 HTML 混合連接（去重後連接）
    norm_t1 = re.sub(r"\s+", " ", t1).strip()
    norm_t2 = re.sub(r"\s+", " ", t2).strip()
    if norm_t2 in norm_t1:
        return t1
    if norm_t1 in norm_t2:
        return t2

    # 段落連接
    if "<table" in t1 or "<table" in t2 or "<ul" in t1 or "<ul" in t2:
        return sanitize_and_balance_html(f"{t1}\n{t2}")

    return f"{t1} {t2}".strip()


# ── 題號前綴剝除工具 ──────────────────────────────────────────────────────────

def strip_question_id_prefix(question_id: str, question_text: str) -> str:
    """
    從 question_text 開頭剝除任何殘留的題號前綴（支援純文字或 HTML 包裝）。
    例如：
      - qid="1.1", text="1.1 Please provide..." -> "Please provide..."
      - qid="II.1.a", text="II.1.a. Outline of..." -> "Outline of..."
      - qid="1.1", text="<p>1.1 Please provide...</p>" -> "<p>Please provide...</p>"
      - qid="1.1", text="<p><strong>1.1.</strong> Please provide...</p>" -> "<p>Please provide...</p>"
      - qid="1.1.2", text="(2) What is..." -> "What is..."
      - qid="Part A.1", text="Part A.1 - Description" -> "Description"
    """
    if not question_text or not question_text.strip():
        return ""

    text = question_text.strip()

    # 1. 偵測開頭的 HTML 標籤（如 <p>, <div>, <span>）
    prefix_html_match = re.match(r"^((?:<\s*(?:p|div|span)[^>]*>\s*)+)", text, re.IGNORECASE)
    prefix_html = prefix_html_match.group(1) if prefix_html_match else ""
    body = text[len(prefix_html):].strip()

    # 2. 如果 body 開頭包了 <strong>/<b>/<em>/<u> 標籤且內部為題號，先清理該內層標籤
    strong_prefix_match = re.match(r"^(?:<\s*(strong|b|em|u)[^>]*>)\s*([^<]+)\s*(?:<\s*/\s*\1\s*>)\s*", body, re.IGNORECASE)
    if strong_prefix_match:
        inner_content = strong_prefix_match.group(2).strip()
        is_id = False
        if question_id and (question_id.lower() in inner_content.lower() or inner_content.lower() in question_id.lower()):
            is_id = True
        elif re.match(r"^(?:[0-9IVXLCDM]+(?:\.[0-9A-Za-z]+)*|[A-Z]|\([A-Za-z0-9]+\))[:.\-、\s]*$", inner_content, re.IGNORECASE):
            is_id = True
        elif re.match(r"^(?:Q|Question|No\.|Item)\s*[0-9A-Za-z.]*", inner_content, re.IGNORECASE):
            is_id = True

        if is_id:
            body = body[strong_prefix_match.end():].strip()

    # 3. 建立基於 question_id 的候選正規表達式
    patterns = []
    if question_id:
        qid_clean = question_id.strip()
        escaped_qid = re.escape(qid_clean)
        # 精確比對完整 qid (如 "II.1.a", "Part A.1", "1.1.2")
        patterns.append(rf"^{escaped_qid}(?:[:.\-、\s]+|\s+)")

        # 比對去除前綴關鍵字後的簡短題號 (如 "Part A.1" -> "A.1", "Chapter II.1" -> "II.1")
        simplified_qid = re.sub(r"^(?:Part|Chapter|Section|Annex|Appendix)\s+", "", qid_clean, flags=re.IGNORECASE).strip()
        if simplified_qid and simplified_qid != qid_clean:
            patterns.append(rf"^{re.escape(simplified_qid)}(?:[:.\-、\s]+|\s+)")

        # 比對題號末節編號 (如 qid="1.1.2" 結尾為 "2", 匹配 "(2)", "(2).", "2.", "2. ")
        tail_part = qid_clean.split(".")[-1].strip()
        if tail_part:
            patterns.append(rf"^\({re.escape(tail_part)}\)(?:[:.\-、\s]+|\s*)")
            patterns.append(rf"^{re.escape(tail_part)}\.(?:[:.\-、\s]+|\s*)")

    # 4. 通用題號前綴 (如 "Q1:", "Question 1.1:", "No. 1.2:")
    patterns.append(r"^(?:Q\b|Question\b|No\.|Item\b)\s*[:.\-、\s]*[A-Za-z0-9.]*[:.\-、\s]+")
    # 通用括號題號前綴 (如 "(a) ", "(1) ", "(i) ")
    patterns.append(r"^\([A-Za-z0-9]+\)\s+")
    # 通用數字/羅馬數字點號前綴 (如 "1. ", "II.1. ", "1.1.2. ")
    patterns.append(r"^(?:[0-9]+|[IVXLCDM]+)(?:\.[0-9A-Za-z]+)*\.\s+")

    for pat in patterns:
        m = re.match(pat, body, re.IGNORECASE)
        if m:
            body = body[m.end():].strip()
            break

    # 5. 去除剝除後殘留在開頭的多餘標點（如冒號、頓號、破折號）
    body = re.sub(r"^[:.\-、\s]+", "", body).strip()

    if not body:
        return ""

    return f"{prefix_html}{body}"


# ── 純填答空格與留白底線過濾工具 ──────────────────────────────────────────────

def strip_fill_in_blanks(text: str) -> str:
    """
    濾除 question_text 中純填答留白的空格、底線、點線與純回覆填寫框。
    例如：
      - <p>[________________________________________________________________________________]</p> -> ""
      - <p>[____________________]</p> -> ""
      - Please specify: __________________________ -> Please specify:
      - Details: [                    ] -> Details:
      - Answer: [________________] -> ""
    """
    if not text:
        return ""

    cleaned = text

    # 1. 移除整段式純填答 HTML 區塊（例如 <p>[_____]</p>, <p>_____</p>, <div>[   ]</div>）
    blank_block_re = re.compile(
        r"<\s*(?:p|div|span)[^>]*>\s*(?:\[\s*(?:_{2,}|\s{2,}|\.{3,}|─{2,}|━{2,}|—{2,})\s*\]|_{3,}|\.{4,}|─{3,}|━{3,}|—{3,}|\[\s{3,}\]|(?:Answer|Reply|Comments?|Notes?):\s*(?:\[\s*_{0,}\s*\]|_{2,}|\.{3,}))\s*<\s*/\s*(?:p|div|span)\s*>",
        re.IGNORECASE
    )
    cleaned = blank_block_re.sub("", cleaned)

    # 2. 移除行內純填答底線、長括號空格或點線
    cleaned = re.sub(r"\[\s*_{2,}\s*\]", "", cleaned)
    cleaned = re.sub(r"\[\s{3,}\]", "", cleaned)
    cleaned = re.sub(r"_{3,}", "", cleaned)
    cleaned = re.sub(r"─{3,}|━{3,}|—{3,}", "", cleaned)
    cleaned = re.sub(r"\.{4,}", "", cleaned)

    # 3. 移除末尾殘留的 "Answer: ", "Reply: " 等提示詞
    cleaned = re.sub(r"(?:<p>\s*)?(?:Answer|Reply)\s*:\s*(?:</p>)?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # 4. 清理因刪除而產生的空 HTML 標籤（如 <p></p>, <div>   </div>）
    cleaned = re.sub(r"<\s*(?:p|div|span)[^>]*>\s*<\s*/\s*(?:p|div|span)\s*>", "", cleaned, flags=re.IGNORECASE)

    # 5. 清理多餘連續空行
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

    return cleaned.strip()


# ── 主公開介面 ────────────────────────────────────────────────────────────────

def format_composite_question_text(raw_text: str, question_id: str = "") -> str:
    """
    全域容錯後處理器：
    1. 濾除純填答底線與空白填寫框（如 <p>[_____]</p>）。
    2. 剝除 question_text 開頭殘留的題號前綴。
    3. 若包含未轉 HTML 的 Markdown 表格，自動轉為 <table>。
    4. 若包含未轉 HTML 的 Markdown 清單，自動轉為 <ul>/<li>。
    5. 執行標籤平衡與安全清理。
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text.strip()

    # 0. 濾除純填答底線與空白填寫框
    text = strip_fill_in_blanks(text)

    # 1. 剝除開頭題號前綴
    text = strip_question_id_prefix(question_id, text)

    # 2. 偵測並轉換 Markdown Pipe 表格
    if "|" in text and "<table" not in text.lower():
        text = convert_markdown_table_to_html(text)

    # 3. 偵測並轉換 Markdown 清單 / 勾選單
    if re.search(r"(?:^|\n)\s*(?:[-*•]|\d+\.)\s+", text) and "<ul" not in text.lower() and "<ol" not in text.lower():
        text = convert_markdown_checklist_to_html(text)

    # 4. 再次清除可能因轉換留下的空白段落並平衡標籤
    text = strip_fill_in_blanks(text)
    return sanitize_and_balance_html(text)
