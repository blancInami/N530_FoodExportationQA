"""
模組三：防幻覺反向驗證機制

功能：
    對 LLM 回傳的每一組雙語詞彙對執行「反向存在性驗證」：
    確認 chinese_term 確實出現在中文原始全文中，
    english_term 確實出現在英文原始全文中。
    未通過驗證的詞彙對直接捨棄，以阻斷模型幻覺進入最終結果。

核心設計：
    比對前先對原始文本執行 clean_markdown() 清洗，
    剔除 Markdown 語法符號、特殊空白等干擾字元，
    提高字串比對的容錯率（避免因格式符號導致漏判）。

防幻覺三層防線中本模組負責第三層（反向子字串驗證），
其餘兩層（Prompt 約束 / Temperature=0）已在 llm_client.py 中實作。
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


# ── 文本清洗函式 ──────────────────────────────────────────────────────────────

def clean_markdown(text: str) -> str:
    """
    清洗 Markdown 文本，去除格式符號與多餘空白，以利子字串比對。

    清洗步驟（依序執行）：
    1. 移除 Markdown 標題符號（行首 # 符號）
    2. 移除強調符號（** / * / __ / _）
    3. 移除連結語法 [text](url) → 保留 text
    4. 移除圖片語法 ![alt](url)
    5. 移除 HTML 標籤（如 <br>、<p>）
    6. 移除表格分隔線（| --- | --- |）
    7. 移除行內程式碼反引號（`code`）
    8. 移除零寬字元與不斷行空格等特殊 Unicode 字元
    9. 壓縮連續空白（多個空格/換行/Tab → 單一空格）

    Args:
        text: 原始 Markdown 文字

    Returns:
        清洗後的純文字字串，保留原始詞彙但去除格式符號。
    """
    if not text:
        return ""

    # 步驟 1：移除行首的 Markdown 標題符號（# ## ### 等）
    result = re.sub(r"(?m)^#{1,6}\s*", "", text)

    # 步驟 2：移除粗體與斜體符號（** / * / __ / _）
    # 先處理雙符號，再處理單符號，避免多餘殘留
    result = re.sub(r"\*{2}|_{2}", "", result)
    result = re.sub(r"\*|_", "", result)

    # 步驟 3：移除 Markdown 連結，保留顯示文字 [text](url) → text
    result = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", result)

    # 步驟 4：移除圖片語法 ![alt](url)
    result = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", result)

    # 步驟 5：移除 HTML 標籤（如 <br>、<strong>）
    result = re.sub(r"<[^>]+>", "", result)

    # 步驟 6：移除表格分隔行（| --- | :---: | 等）
    result = re.sub(r"(?m)^\|[\s\-:|]+\|.*$", "", result)

    # 步驟 7：移除行內程式碼反引號（`code`）
    result = re.sub(r"`[^`]*`", "", result)

    # 步驟 8：移除表格的管線符號 |（用於分欄）
    result = result.replace("|", " ")

    # 步驟 9：移除特殊 Unicode 字元
    # - 零寬字元（U+200B、U+200C、U+200D、U+FEFF 等）
    # - 不斷行空格（U+00A0）
    # - 其他格式類 Unicode 字元（category Cf）
    result = "".join(
        ch for ch in result
        if unicodedata.category(ch) not in {"Cf"}  # Cf = Format characters
        and ch not in {"\u200b", "\u200c", "\u200d", "\ufeff", "\u00a0"}
    )

    # 步驟 10：壓縮連續空白（空格、Tab、換行）為單一空格
    result = re.sub(r"\s+", " ", result).strip()

    return result


# ── 單一詞對驗證 ──────────────────────────────────────────────────────────────

def validate_term_pair(
    zh_term: str,
    en_term: str,
    zh_full_text: str,
    en_full_text: str,
) -> bool:
    """
    驗證一組雙語詞彙對是否確實存在於對應的原始全文中。

    驗證規則：
    - 中文詞：zh_term 必須是清洗後中文全文的子字串（大小寫不影響）
    - 英文詞：en_term 必須是清洗後英文全文的子字串（不分大小寫）
    - 兩個條件皆須成立，才視為通過驗證

    為何使用全文（而非 chunk）：
        避免因 chunk 切割邊界恰好截斷詞彙，導致正確詞彙被誤判為幻覺。

    Args:
        zh_term      : LLM 回傳的中文專有名詞
        en_term      : LLM 回傳的英文專有名詞
        zh_full_text : 中文 PDF 的完整萃取文字（未切塊）
        en_full_text : 英文 PDF 的完整萃取文字（未切塊）

    Returns:
        True：兩個詞彙均在對應全文中找到 → 保留
        False：任一詞彙不在全文中 → 判定為幻覺，捨棄
    """
    # 對原始全文進行 Markdown 清洗（此操作只需在外部做一次，
    # 但為保持函式獨立性，此處每次都執行；
    # 如需效能優化，可在呼叫端預先清洗後傳入）
    clean_zh = clean_markdown(zh_full_text)
    clean_en = clean_markdown(en_full_text)

    # 中文子字串比對（中文詞通常不需要大小寫處理，但仍做統一比對）
    zh_found = zh_term.strip() in clean_zh

    # 英文子字串比對（不分大小寫）
    en_found = en_term.strip().lower() in clean_en.lower()

    if not zh_found:
        logger.debug("中文詞未在全文中找到，判定為幻覺：'%s'", zh_term)
    if not en_found:
        logger.debug("英文詞未在全文中找到，判定為幻覺：'%s'", en_term)

    return zh_found and en_found


# ── 批次過濾 ──────────────────────────────────────────────────────────────────

def filter_hallucinations(
    terms: list[dict],
    zh_full_text: str,
    en_full_text: str,
) -> list[dict]:
    """
    批次過濾 LLM 回傳的詞彙列表，移除未通過反向驗證的幻覺詞彙。

    此函式會對 zh_full_text / en_full_text 預先清洗一次，
    再對每個詞彙對呼叫驗證，以避免重複清洗的效能開銷。

    Args:
        terms        : LLM 回傳並經過格式解析的詞彙列表
                       每個 dict 含 'chinese_term' 與 'english_term'
        zh_full_text : 中文 PDF 的完整萃取文字（未切塊）
        en_full_text : 英文 PDF 的完整萃取文字（未切塊）

    Returns:
        通過驗證的詞彙列表（子集）
    """
    if not terms:
        return []

    # 預先清洗全文（只做一次，供後續所有詞彙比對使用）
    clean_zh = clean_markdown(zh_full_text)
    clean_en = clean_markdown(en_full_text)

    valid: list[dict] = []
    rejected_count = 0

    for item in terms:
        zh_term = item.get("chinese_term", "").strip()
        en_term = item.get("english_term", "").strip()

        # 跳過空值（理論上 llm_client._validate_term_list 已過濾，此為雙重保險）
        if not zh_term or not en_term:
            rejected_count += 1
            continue

        # 使用預清洗的全文進行比對（不再重複清洗）
        zh_found = zh_term in clean_zh
        en_found = en_term.lower() in clean_en.lower()

        if zh_found and en_found:
            valid.append(item)
        else:
            rejected_count += 1
            logger.debug(
                "幻覺詞彙已捨棄：中文='%s'（%s）  英文='%s'（%s）",
                zh_term, "✓" if zh_found else "✗",
                en_term, "✓" if en_found else "✗",
            )

    logger.info(
        "反向驗證完成：輸入 %d 筆  通過 %d 筆  捨棄 %d 筆",
        len(terms), len(valid), rejected_count,
    )
    return valid
