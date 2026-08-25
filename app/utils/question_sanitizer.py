"""
Question Sanitizer & Defense Utility
提供問卷題目清洗、JSON 陣列修補、題號繼承修復、欄位標頭雜訊過濾、表格融合與語義去重工具。
"""
from __future__ import annotations

import json
import logging
import re

from app.utils.html_formatter import format_composite_question_text, merge_html_composite_texts

logger = logging.getLogger(__name__)

_PURE_COLUMN_HEADERS = {
    "no", "no.", "item", "item no", "item no.", "question", "questions",
    "question / description", "description", "details", "competent authority response",
    "response", "answer", "remarks", "remarks / comments", "comments",
    "signature", "signature / date", "date", "official use only", "score",
}

_DISCLAIMER_PATTERNS = [
    r"^(?:Please\s+complete\s+in\s+English|Please\s+use\s+block\s+capitals)\b.*",
    r"^(?:This\s+questionnaire\s+is\s+established\s+under|In\s+accordance\s+with\s+Regulation\s+\()\b.*",
    r"^(?:Note\s*:\s*(?:Please\s+fill|Answers\s+should|All\s+questions|Confidential))\b.*",
    r"^(?:General\s+Instructions?|Guidance\s+Notes?)\s*:\s*.*",
]


def clean_json(raw: str) -> str:
    """清理 Markdown code fence 標籤。"""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned.strip()


def repair_truncated_json_array(raw_str: str) -> list[dict]:
    """
    【斷尾 JSON 陣列自動修復器】
    當 VLM 輸出因達到 Token 上限而在末端中途中斷（例如 Unterminated string 或 Expecting property name）時，
    自動尋找最後一個完整閉合的物件 '}' 並補上 ']'，100% 救回前面所有已成功辨識的大綱節點。
    """
    if not raw_str or not raw_str.strip():
        return []

    cleaned = clean_json(raw_str)

    # 1. 嘗試直接完整解析
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [it for it in data if isinstance(it, dict)]
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    # 2. 尋找最後一個完整閉合的 '}'
    last_brace_idx = cleaned.rfind("}")
    first_bracket_idx = cleaned.find("[")

    if last_brace_idx != -1 and first_bracket_idx != -1 and last_brace_idx > first_bracket_idx:
        truncated_slice = cleaned[first_bracket_idx : last_brace_idx + 1] + "]"
        try:
            repaired_data = json.loads(truncated_slice)
            if isinstance(repaired_data, list):
                logger.info("[LangGraph JSON修復] 成功修復斷尾 JSON 陣列，救回 %d 個大綱節點", len(repaired_data))
                return [it for it in repaired_data if isinstance(it, dict)]
        except Exception:
            pass

    # 3. 若上述結構修復仍失敗，使用獨立物件正則逐一抽樣提取
    extracted_objects: list[dict] = []
    object_matches = re.finditer(r"\{[^{}]*\}", cleaned)
    for m in object_matches:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and (obj.get("section_id") or obj.get("id") or obj.get("title")):
                extracted_objects.append(obj)
        except Exception:
            continue

    if extracted_objects:
        logger.info("[LangGraph JSON修復] 透過獨立物件正規表達式救回 %d 個大綱節點", len(extracted_objects))
        return extracted_objects

    return []


def validate_extracted_questions(data: object) -> list[dict]:
    """驗證並清洗萃取出的題目清單。"""
    if not isinstance(data, list):
        return []
    validated: list[dict] = []
    for item in data:
        if isinstance(item, dict) and "question_id" in item and "question_text" in item:
            qid = str(item["question_id"]).strip()
            raw_qtext = str(item["question_text"]).strip()
            qtext = format_composite_question_text(raw_qtext, question_id=qid)
            if qid and qtext:
                q_dict = {"question_id": qid, "question_text": qtext}
                if "section_id" in item and item["section_id"]:
                    q_dict["section_id"] = str(item["section_id"]).strip()
                validated.append(q_dict)
    return validated


def normalize_single_question_id(raw_id: str) -> str:
    """單題號基礎標點與符號清洗。"""
    qid = str(raw_id).strip()
    qid = re.sub(r"\b(Part|Chapter|Section|Annex|Appendix)_([A-Za-z0-9IVXLCDM]+)\b", r"\1 \2", qid, flags=re.IGNORECASE)
    qid = re.sub(r"_+", ".", qid)
    qid = re.sub(r"\.+", ".", qid)
    qid = re.sub(r"\s*\.\s*", ".", qid)
    qid = qid.strip(". ").strip()
    qid = re.sub(r"\b(Part\s+[A-Za-z0-9IVXLCDM]+)\s+(\d+)\b", r"\1.\2", qid, flags=re.IGNORECASE)
    return qid


def filter_false_positive_items(items: list[dict]) -> list[dict]:
    """
    【防護層 2：偽題目與版面結構雜訊全域過濾器】
    徹底剔除：
    1. 獨立的表格欄位標頭（如 'No.', 'Question', 'Remarks', 'Competent Authority Response'）。
    2. 純填寫說明、法規引文或免責宣告。
    3. 純空白填答框（如 <p>[___]</p>）或無實質內容的空表格。
    """
    if not items:
        return []

    clean_items: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("question_id", "")).strip()
        qtext = str(it.get("question_text", "")).strip()

        if not qid or not qtext:
            continue

        # 1. 檢查題號與題幹是否僅為純表格欄位名稱 (Column Headers)
        norm_qid = re.sub(r"[\s\.:_\-]+", " ", qid).strip().lower()
        norm_text = re.sub(r"<[^>]+>", " ", qtext).strip().lower()
        norm_text_clean = re.sub(r"[\s\.:_\-]+", " ", norm_text).strip()

        if norm_qid in _PURE_COLUMN_HEADERS and (not norm_text_clean or norm_text_clean in _PURE_COLUMN_HEADERS):
            continue
        if norm_text_clean in _PURE_COLUMN_HEADERS and len(norm_text_clean) < 35:
            # 確保不是有實質表格的題目
            if "<table" not in qtext.lower() and "<ul" not in qtext.lower():
                continue

        # 2. 檢查是否為純填寫指引或法規引文宣告 (Disclaimers & Guidance)
        is_disclaimer = False
        for pat in _DISCLAIMER_PATTERNS:
            if re.match(pat, norm_text, re.IGNORECASE):
                if "<table" not in qtext.lower() and "?" not in qtext:
                    is_disclaimer = True
                    break
        if is_disclaimer:
            continue

        # 3. 檢查是否為純空白填寫底線或空內容
        stripped_text = re.sub(r"<[^>]+>", "", qtext).strip()
        if not stripped_text and "<table" not in qtext.lower() and "<ul" not in qtext.lower():
            continue

        clean_items.append(it)

    return clean_items


def repair_nested_subquestion_hierarchy(items: list[dict], default_parent_id: str = "") -> list[dict]:
    """
    【防護層 3：多層級子題號繼承修補器】
    解決遇到 (a), (b), (i), (ii), (1) 孤立子題號時丟失父層定義（如 1.1 或 II.1）之問題。
    自動將孤立題號展開為標準扁平化題號（例如 1.1 + (a) -> 1.1.a, 1.1.a + (i) -> 1.1.a.i）。
    """
    if not items:
        return []

    repaired: list[dict] = []
    root_parent_id = default_parent_id.strip()
    alpha_parent_id = ""

    for it in items:
        raw_qid = str(it.get("question_id", "")).strip()
        q_copy = dict(it)

        # 1. 檢查羅馬數字子子題 (i), (ii), (iii), (iv), (v)
        roman_match = re.match(r"^\(?([ivxlcdm]+)\)?$", raw_qid, re.IGNORECASE)
        # 2. 檢查字母子題 (a), (b), (c)...
        alpha_match = re.match(r"^\(?([a-zA-Z])\)?$", raw_qid)
        # 3. 檢查數字子題 (1), (2), (3)...
        digit_match = re.match(r"^\(?(\d+)\)?$", raw_qid)

        is_roman = bool(roman_match and roman_match.group(1).lower() in ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"])
        is_alpha = bool(alpha_match and not is_roman)
        is_digit = bool(digit_match and len(raw_qid) <= 3 and "." not in raw_qid)

        if is_roman and (alpha_parent_id or root_parent_id):
            parent = alpha_parent_id if alpha_parent_id else root_parent_id
            sub_clean = roman_match.group(1).lower()
            repaired_qid = f"{parent}.{sub_clean}"
            q_copy["question_id"] = repaired_qid
        elif is_alpha and root_parent_id:
            sub_clean = alpha_match.group(1).lower()
            repaired_qid = f"{root_parent_id}.{sub_clean}"
            q_copy["question_id"] = repaired_qid
            alpha_parent_id = repaired_qid
        elif is_digit and root_parent_id and ("." not in raw_qid):
            sub_clean = digit_match.group(1)
            repaired_qid = f"{root_parent_id}.{sub_clean}"
            q_copy["question_id"] = repaired_qid
        else:
            norm_qid = normalize_single_question_id(raw_qid)
            root_parent_id = norm_qid
            alpha_parent_id = ""
            q_copy["question_id"] = norm_qid

        repaired.append(q_copy)

    return repaired


def detect_and_convert_weak_subsections(items: list[dict], heading_stack: list[str]) -> tuple[list[dict], list[str]]:
    """
    【防護層 6：弱特徵小節標題動態探測器】
    辨識排版弱反差的小節大標題（例如 2.1.3 Storage conditions），
    自動將其從小節題目中剔除，並動態提升為標題棧（Heading Stack）節點，避免誤當題目輸出。
    """
    if not items:
        return [], heading_stack

    filtered_questions: list[dict] = []
    updated_stack = list(heading_stack)

    for it in items:
        qid = str(it.get("question_id", "")).strip()
        qtext = str(it.get("question_text", "")).strip()

        # 檢驗是否符合小節標題特徵：
        # 1. 無 HTML 表格、無勾選清單、無問號
        has_table_or_list = bool(re.search(r"<(?:table|ul|ol)\b", qtext, re.IGNORECASE))
        has_question_mark = "?" in qtext
        # 2. 文本長度短（< 60 字元），且為純名詞片語（無問答動詞 Please / Describe / Is / Are）
        has_question_verb = bool(re.search(
            r"\b(?:please|indicate|provide|describe|specify|state|answer|explain|are|is|do|does|how|what|which)\b",
            qtext,
            re.IGNORECASE,
        ))

        is_heading_candidate = (
            not has_table_or_list
            and not has_question_mark
            and not has_question_verb
            and len(qtext) < 60
            and re.match(r"^[A-Z][a-zA-Z0-9\s\-_/,&]+$", qtext.strip())
        )

        # 題號符合小節編號格式（如 2.1, 2.1.3, B.2）
        is_sec_id_pattern = bool(re.match(r"^(?:[A-Za-z0-9IVXLCDM]+\.)+\d+$", qid) or re.match(r"^[A-Za-z]\.\d+$", qid))

        if is_heading_candidate and is_sec_id_pattern:
            # 判定為小節標題，加入標題棧
            heading_title = f"{qid} {qtext}".strip()
            if heading_title not in updated_stack:
                updated_stack.append(heading_title)
        else:
            filtered_questions.append(it)

    return filtered_questions, updated_stack


def canonicalize_all_question_ids(items: list[dict]) -> list[dict]:
    """全局題號正規化與前綴校正。"""
    if not items:
        return []

    # 1. 偽題目與純欄位標頭過濾
    clean_raw = filter_false_positive_items(items)
    # 2. 多層級子題號繼承修補
    repaired_items = repair_nested_subquestion_hierarchy(clean_raw)

    first_pass: list[dict] = []
    for it in repaired_items:
        raw_id = it.get("question_id", "")
        raw_txt = it.get("question_text", "")
        norm_id = normalize_single_question_id(raw_id)
        if norm_id and raw_txt:
            first_pass.append({"question_id": norm_id, "question_text": raw_txt})

    has_part_prefix_count = sum(1 for it in first_pass if re.match(r"^Part\s+[A-Za-z0-9IVXLCDM]+", it["question_id"], flags=re.IGNORECASE))
    total_items = len(first_pass)
    prefer_part_prefix = total_items > 0 and (has_part_prefix_count / total_items >= 0.3)

    aligned_items: list[dict] = []
    for it in first_pass:
        qid = it["question_id"]
        if prefer_part_prefix:
            m = re.match(r"^([A-Za-z])\.(\d.*)$", qid)
            if m:
                qid = f"Part {m.group(1).upper()}.{m.group(2)}"
        aligned_items.append({"question_id": qid, "question_text": it["question_text"]})

    return deduplicate_items(aligned_items)


def is_prompt_expecting_table(text: str) -> bool:
    """檢查題幹是否預期隨附表格。"""
    cleaned = re.sub(r"<[^>]+>", " ", text).strip().lower()
    return bool(re.search(
        r"(?:table\s+below|following\s+table|in\s+the\s+table|as\s+follows|provide\s+details\s+in\s+table|statistics\s+in\s+table|:\s*$)",
        cleaned,
    ))


def is_orphan_table_or_list_item(item: dict, prev_item: dict) -> bool:
    """檢查某條目是否為孤立的 HTML 表格/清單（且前一題預期表格或同題無題幹）。"""
    txt = item.get("question_text", "").strip()
    non_html = re.sub(r"<[^>]+>", "", txt).strip()
    has_table = bool(re.search(r"<(?:table|ul|ol)\b", txt, re.IGNORECASE))
    if not has_table:
        return False
    prev_txt = prev_item.get("question_text", "").strip()
    if is_prompt_expecting_table(prev_txt):
        return True
    if len(non_html) < 5 and "<table" not in prev_txt:
        return True
    return False


def deduplicate_items(items: list[dict]) -> list[dict]:
    """
    【防護層 4：同題連續表格/選項無縫融合與去重器】
    """
    deduped: list[dict] = []
    seen_ids: dict[str, int] = {}

    for item in items:
        qid = item["question_id"]
        qtext = item["question_text"]

        if deduped and is_orphan_table_or_list_item(item, deduped[-1]):
            prev_item = deduped[-1]
            prev_text = prev_item["question_text"]
            merged = merge_html_composite_texts(prev_text, qtext)
            prev_item["question_text"] = merged
            continue

        if qid not in seen_ids:
            seen_ids[qid] = len(deduped)
            deduped.append({"question_id": qid, "question_text": qtext})
        else:
            existing_idx = seen_ids[qid]
            is_adjacent = (existing_idx == len(deduped) - 1)
            existing_text = deduped[existing_idx]["question_text"]

            if is_adjacent:
                merged_text = merge_html_composite_texts(existing_text, qtext)
                deduped[existing_idx]["question_text"] = merged_text
            else:
                if qtext.strip() not in existing_text:
                    alt_qid = f"{qid}.2"
                    seen_ids[alt_qid] = len(deduped)
                    deduped.append({"question_id": alt_qid, "question_text": qtext})

    return deduped
