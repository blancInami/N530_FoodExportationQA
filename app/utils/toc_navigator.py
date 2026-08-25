"""
TOC & Section Topological Navigator Utility
提供問卷章節階層樹建構、實體頁碼排序、目錄錨點校準、題號前綴歸屬與閱讀工作記憶生成工具。
"""
from __future__ import annotations

import logging
import re

from app.schemas.langgraph import OutlineItem, TocAnchor
from app.services.llm import chat_completion
from app.utils.prompt_loader import get_langgraph_prompt
from app.utils.question_sanitizer import normalize_single_question_id

logger = logging.getLogger(__name__)


def split_section_id_and_depiction(
    raw_id: str,
    raw_title: str = "",
    raw_depiction: str = "",
) -> tuple[str, str]:
    """
    【語義化章節純化器】
    信任 LLM 語義分離結果，安全純化區塊 key 與領域前言說明：
    - clean_id: 純化為簡潔代碼（如 "B", "Part A", "Chapter I", "1"）。
    - clean_depiction: 整合標題領域說明（如 "Competent authority(ies)"）與前言段落。
    """
    id_str = str(raw_id or "").strip()
    title_str = str(raw_title or "").strip()
    dep_str = str(raw_depiction or "").strip()

    # 1. 處理帶有分隔符的原始 ID (例如 "B: General Information" 或 "C - Competent authority")
    if not title_str:
        for delim in [":", " - ", " – ", " — "]:
            if delim in id_str:
                parts = id_str.split(delim, 1)
                id_str = parts[0].strip()
                title_str = parts[1].strip()
                break

    clean_id = normalize_single_question_id(id_str)
    if ":" in clean_id:
        clean_id = clean_id.split(":", 1)[0].strip()

    # 2. 若 title_str 開頭仍含有 clean_id 前綴（例如 "B - General Information"），安全剝除 "B - "（使用詞界保護首字）
    if title_str and clean_id:
        escaped_id = re.escape(clean_id)
        title_str = re.sub(
            rf"^(?:Part\s+{escaped_id}|Chapter\s+{escaped_id}|Section\s+{escaped_id}|{escaped_id})\b(?:\s*[-–—:.]+\s*|\s+)",
            "",
            title_str,
            flags=re.IGNORECASE,
        ).strip()

    # 3. 組合領域標題與前言說明
    dep_parts: list[str] = []
    if title_str:
        dep_parts.append(title_str)
    if dep_str:
        if not title_str or title_str.lower() not in dep_str.lower():
            dep_parts.append(dep_str)
        elif dep_str.strip() != title_str.strip():
            dep_parts = [dep_str]

    raw_dep_text = "\n\n".join(dep_parts).strip()
    clean_depiction = sanitize_depiction(raw_dep_text, clean_id)
    if not clean_id:
        clean_id = "General"

    return clean_id, clean_depiction


def sanitize_depiction(
    raw_dep: str,
    section_id: str = "",
    questions: list[dict] | None = None,
) -> str:
    """
    【輕量守門員：Depiction 撞題過濾、詞界前綴清理與去重】
    1. 撞題比對：確保題目問句題幹不誤入前言。
    2. 詞界前綴清理：若行首含有章節代號前綴（如 'D - Notifications...'），去除 'D - '，保護首字母。
    3. 多行去重：去除完全重複的標題行或子集行。
    """
    if not raw_dep or not raw_dep.strip():
        return ""

    q_texts_normalized: list[str] = []
    if questions:
        for q in questions:
            q_raw = str(q.get("question_text", "")).strip()
            q_clean = re.sub(r"<[^>]+>", " ", q_raw).strip()
            if q_clean:
                q_texts_normalized.append(re.sub(r"\s+", " ", q_clean.lower()))

    lines = [line.strip() for line in raw_dep.split("\n") if line.strip()]
    clean_lines: list[str] = []
    seen_lower: set[str] = set()

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # 1. 撞題比對（若此行包含本區塊題目的問句題幹，予以剔除）
        line_norm = re.sub(r"\s+", " ", line_clean.lower())
        matched_question = False
        for qt in q_texts_normalized:
            if len(line_norm) >= 15 and (line_norm in qt or qt in line_norm or line_norm[:30] == qt[:30]):
                matched_question = True
                break
        if matched_question:
            continue

        # 2. 去除章節代號前綴（例如 "D - Notifications..." -> "Notifications..."），使用 \b 保護單字首字
        if section_id:
            sec_norm = normalize_single_question_id(section_id)
            escaped_sec_norm = re.escape(sec_norm)
            escaped_sec_id = re.escape(section_id)
            line_clean = re.sub(
                rf"^(?:Part\s+{escaped_sec_norm}|Chapter\s+{escaped_sec_norm}|Section\s+{escaped_sec_norm}|{escaped_sec_norm}|{escaped_sec_id})\b(?:\s*[-–—:.]+\s*|\s+)",
                "",
                line_clean,
                flags=re.IGNORECASE,
            ).strip()
            if not line_clean:
                continue

        line_lower = line_clean.lower()

        # 3. 去除完全重複行或已包含的完全子集段落
        is_duplicate = False
        for seen in seen_lower:
            if line_lower == seen or (len(line_lower) > 10 and (line_lower in seen or seen in line_lower)):
                is_duplicate = True
                break

        if not is_duplicate:
            seen_lower.add(line_lower)
            clean_lines.append(line_clean)

    return "\n\n".join(clean_lines).strip()


def match_section_name(active_chap: str, section_order: list[str]) -> str:
    """
    精準比對當前章節 ID 與 section_order 中的純化區塊代碼。
    避免 'Chapter_I' 誤匹配為 'Chapter_II'，且未匹配時絕不盲目 fallback 污染最後一個章節。
    """
    if not section_order or not active_chap:
        return ""

    act = active_chap.strip().lower()
    act_norm = normalize_single_question_id(active_chap).strip().lower()

    for sn in reversed(section_order):
        sn_norm = normalize_single_question_id(sn).strip().lower()
        if sn.lower() == act or sn_norm == act_norm:
            return sn

    for sn in reversed(section_order):
        sn_clean = sn.replace("_", " ").lower()
        act_clean = act.replace("_", " ").lower()
        if re.match(rf"^{re.escape(act_clean)}$", sn_clean, re.IGNORECASE):
            return sn
        if re.match(rf"^{re.escape(act_clean)}(?:\s*:|\s+|$)", sn_clean, re.IGNORECASE):
            return sn
        if re.match(rf"^(?:Part|Chapter|Section)\s+{re.escape(sn_clean)}$", act_clean, re.IGNORECASE):
            return sn
        if re.match(rf"^(?:Part|Chapter|Section)\s+{re.escape(act_clean)}$", sn_clean, re.IGNORECASE):
            return sn

    for sn in reversed(section_order):
        if ":" in sn:
            sn_id = sn.split(":", 1)[0].strip().lower()
            if sn_id == act or sn_id == act_norm:
                return sn

    return ""


def extract_section_prefix_from_qid(qid: str, known_sections: list[str]) -> str | None:
    """
    從題號字串中解析出所屬區塊（確定性匹配）：
    優先比對較長前綴（如 'C.A', 'C.B' 優於 'C'），杜絕多層複合章節被錯誤匹配為上層單字。
    """
    if not qid or not known_sections:
        return None

    raw_qid = qid.strip()
    norm_qid = normalize_single_question_id(raw_qid)

    # 依照字串長度降冪排序，確保 C.A、C.B 等長標識優先於 C 被匹配
    sorted_known = sorted([s for s in known_sections if s and s != "General"], key=lambda s: len(str(s)), reverse=True)

    for sec in sorted_known:
        sec_clean = normalize_single_question_id(sec)
        escaped_sec = re.escape(sec_clean)
        escaped_raw = re.escape(sec)

        patterns = [
            rf"^{escaped_sec}(?:[\.\s\-_:]+[A-Za-z0-9].*|$)",
            rf"^{escaped_raw}(?:[\.\s\-_:]+[A-Za-z0-9].*|$)",
            rf"^(?:Part|Chapter|Section)\s+{escaped_sec}(?:[\.\s\-_:]+[A-Za-z0-9].*|$)",
        ]
        for pat in patterns:
            if re.match(pat, norm_qid, re.IGNORECASE) or re.match(pat, raw_qid, re.IGNORECASE):
                return sec

    m = re.match(r"^([A-Za-z0-9IVXLCDM]+(?:\.[A-Za-z0-9IVXLCDM]+)?)(?:[\.\-_:]+.*|$)", norm_qid)
    if m:
        candidate_code = m.group(1).upper()
        for sec in sorted_known:
            sec_norm = normalize_single_question_id(sec).upper()
            if sec_norm == candidate_code or sec_norm == f"PART {candidate_code}" or sec_norm == f"CHAPTER {candidate_code}":
                return sec

    return None


def resolve_question_section(
    q: dict,
    page_prev_section: str,
    page_new_section: str,
    known_sections: list[str],
    is_after_transition: bool,
) -> str:
    """
    結合題號前綴分析、單題 section_id 與同頁過渡狀態，確定性判定題目所屬區塊。
    """
    qid = str(q.get("question_id", "")).strip()

    # 1. 題號前綴確定性判定（最高優先權）
    matched_by_prefix = extract_section_prefix_from_qid(qid, known_sections)
    if matched_by_prefix:
        return matched_by_prefix

    # 2. 若題目隨附 VLM 標註之 section_id
    if "section_id" in q and q["section_id"]:
        sid = str(q["section_id"]).strip()
        matched_by_sid = match_section_name(sid, known_sections)
        if matched_by_sid:
            return matched_by_sid

    # 3. 依據同頁過渡狀態切分
    target_chap = page_new_section if is_after_transition else page_prev_section
    if target_chap:
        return match_section_name(target_chap, known_sections)

    if known_sections:
        return known_sections[-1]
    return "General"


def build_page_section_map(
    outline_items: list[dict],
    total_pages: int,
) -> tuple[dict[int, list[OutlineItem]], list[OutlineItem], list[str], dict[str, str]]:
    """
    將大綱節點清單解析、拓撲排序並建立：
      - page_section_map: 每頁對應之 OutlineItem 清單（支援同頁多章節/小節）
      - global_outline_tree: 依頁碼與位置排序之全篇大綱
      - section_order: 循序純化後的主章節列表（嚴格遵循實體頁碼與版面順序）
      - section_depictions: 主章節說明字典
    """
    page_map: dict[int, list[OutlineItem]] = {p: [] for p in range(1, total_pages + 1)}
    raw_valid: list[OutlineItem] = []
    section_order: list[str] = []
    section_depictions: dict[str, str] = {}

    def _item_sort_key(it: dict) -> tuple[int, int]:
        raw_p = it.get("page", it.get("page_number", 1))
        try:
            p_val = int(raw_p)
        except (ValueError, TypeError):
            p_val = 1
        pos = str(it.get("position", "top")).lower()
        pos_rank = 0 if pos == "top" else (1 if pos == "middle" else (2 if pos == "bottom" else 3))
        return (p_val, pos_rank)

    sorted_outline = sorted([it for it in outline_items if isinstance(it, dict)], key=_item_sort_key)

    for item in sorted_outline:
        if not isinstance(item, dict):
            continue
        raw_sid = str(item.get("section_id", item.get("id", ""))).strip()
        raw_sub = str(item.get("subsection_id", "")).strip()
        raw_title = str(item.get("title", item.get("heading_title", ""))).strip()
        raw_dep = str(item.get("depiction", "")).strip()
        raw_pos = str(item.get("position", "top")).strip().lower()
        if raw_pos not in ["top", "middle", "bottom", "full_page"]:
            raw_pos = "top"
        is_major = bool(item.get("is_major_section", False))

        raw_p = item.get("page", item.get("page_number", 1))
        try:
            p_int = max(1, min(int(raw_p), total_pages))
        except (ValueError, TypeError):
            p_int = 1

        clean_sid, clean_dep = split_section_id_and_depiction(raw_sid, raw_title if is_major else "", raw_dep)
        if not clean_sid or clean_sid == "General":
            clean_sid = normalize_single_question_id(raw_sid or raw_sub) or "General"

        if not raw_sub:
            raw_sub = clean_sid

        out_item: OutlineItem = {
            "section_id": clean_sid,
            "subsection_id": raw_sub,
            "title": raw_title,
            "page": p_int,
            "position": raw_pos,
            "is_major_section": is_major,
            "depiction": clean_dep if is_major else raw_dep,
        }
        raw_valid.append(out_item)
        page_map[p_int].append(out_item)

        # 守門防禦：嚴格過濾無效與題目級別代號（如 'None', 'G.72', 'II.1.(a)'）
        is_invalid_sec = (
            not clean_sid
            or clean_sid.lower() in ["none", "null", "general", "n/a"]
            or bool(re.search(r"\([a-z0-9]+\)", clean_sid, re.IGNORECASE))
            or bool(re.match(r"^[A-Z]\.\d{2,}$", clean_sid))
            or clean_sid.count(".") >= 2
        )

        if not is_invalid_sec:
            if clean_sid not in section_order:
                section_order.append(clean_sid)
                section_depictions[clean_sid] = clean_dep
            elif clean_dep and not section_depictions.get(clean_sid):
                section_depictions[clean_sid] = clean_dep

    if not section_order:
        section_order = ["General"]
        section_depictions["General"] = ""

    last_known: list[OutlineItem] = []
    for p in range(1, total_pages + 1):
        if page_map[p]:
            last_known = [dict(page_map[p][-1])]  # type: ignore
        elif last_known:
            propagated = dict(last_known[0])
            propagated["page"] = p
            propagated["position"] = "full_page"
            propagated["is_major_section"] = False
            page_map[p] = [propagated]  # type: ignore

    return page_map, raw_valid, section_order, section_depictions


def reorder_sections_by_physical_page(
    outline_tree: list[OutlineItem],
    page_section_map: dict[int, list[OutlineItem]],
    known_sections: list[str],
) -> list[str]:
    """
    根據全篇大綱與各頁章節地圖中「首次出現的實體頁碼與垂直位置」，對章節列表進行嚴格的物理順序排序。
    """
    section_coords: dict[str, tuple[int, int]] = {}

    def _pos_rank(pos: str) -> int:
        p = str(pos or "").lower().strip()
        return 0 if p == "top" else (1 if p == "middle" else (2 if p == "bottom" else 3))

    for it in outline_tree:
        sec = it.get("section_id", "")
        if sec:
            coord = (int(it.get("page", 1)), _pos_rank(it.get("position", "top")))
            if sec not in section_coords or coord < section_coords[sec]:
                section_coords[sec] = coord

    for p_num, sec_list in page_section_map.items():
        for it in sec_list:
            sec = it.get("section_id", "")
            if sec:
                coord = (int(p_num), _pos_rank(it.get("position", "top")))
                if sec not in section_coords or coord < section_coords[sec]:
                    section_coords[sec] = coord

    all_secs = list(dict.fromkeys([s for s in known_sections if s] + list(section_coords.keys())))
    if not all_secs:
        return ["General"]

    max_known_p = max((c[0] for c in section_coords.values()), default=1)

    def _sec_sort_key(sec: str) -> tuple[int, int, int]:
        if sec in section_coords:
            c = section_coords[sec]
            return (c[0], c[1], 0)
        idx = known_sections.index(sec) if sec in known_sections else 999
        return (max_known_p + 1, 0, idx)

    return sorted(all_secs, key=_sec_sort_key)


def calculate_toc_anchors(toc_data: list[dict], total_pages: int) -> list[TocAnchor]:
    """從目錄結構計算起始/結束頁碼錨點。"""
    if not toc_data:
        return []
    valid = [t for t in toc_data if isinstance(t, dict) and (t.get("id") or t.get("section_id"))]
    if not valid:
        return []

    anchors: list[TocAnchor] = []
    for i, item in enumerate(valid):
        cid = str(item.get("section_id", item.get("id", f"Chapter_{i+1}"))).strip()
        title = str(item.get("title", "")).strip()
        raw_p = item.get("page", 1)
        try:
            start_p = max(1, min(int(raw_p), total_pages))
        except (ValueError, TypeError):
            start_p = 1
        anchors.append({
            "id": cid,
            "title": title,
            "start_page": start_p,
            "end_page": total_pages,
        })

    anchors.sort(key=lambda a: a["start_page"])
    for i in range(len(anchors) - 1):
        anchors[i]["end_page"] = max(anchors[i]["start_page"], anchors[i + 1]["start_page"] - 1)
    anchors[-1]["end_page"] = total_pages
    return anchors


def get_active_anchor(anchors: list[TocAnchor], current_page: int | None = None, page_no: int | None = None) -> TocAnchor | None:
    """查詢當前頁面所落之目錄錨點。"""
    p = current_page if current_page is not None else (page_no if page_no is not None else 1)
    for a in anchors:
        if a["start_page"] <= p <= a["end_page"]:
            return a
    return anchors[-1] if anchors else None


def calibrate_toc_anchors(
    anchors: list[TocAnchor],
    current_page: int,
    detected_chapter: dict,
    total_pages: int,
) -> tuple[list[TocAnchor], bool, str]:
    """動態校準目錄錨點與實際物理頁碼。"""
    detected_id = str(detected_chapter.get("id", "")).strip()
    detected_title = str(detected_chapter.get("title", "")).strip()
    if not detected_id:
        return anchors, False, ""

    updated: list[TocAnchor] = []
    matched = False
    milestone_msg = ""

    for a in anchors:
        a_copy = dict(a)
        if a["id"].lower() == detected_id.lower():
            matched = True
            old_start = a["start_page"]
            if old_start != current_page:
                a_copy["start_page"] = current_page
                a_copy["is_calibrated"] = True
                milestone_msg = (
                    f"章節 [{detected_id}] 在第 {current_page} 頁實際出現！"
                    f"calibrated from p.{old_start} -> real physical p.{current_page}"
                )
            if detected_title and not a_copy.get("title"):
                a_copy["title"] = detected_title
        updated.append(a_copy)

    if not matched:
        milestone_msg = f"發現未在目錄登記之新章節 [{detected_id}] 於第 {current_page} 頁！已動態插入大綱樹"
        updated.append({
            "id": detected_id,
            "title": detected_title,
            "start_page": current_page,
            "end_page": total_pages,
            "is_calibrated": True,
        })

    updated.sort(key=lambda a: a["start_page"])
    for i in range(len(updated) - 1):
        updated[i]["end_page"] = max(updated[i]["start_page"], updated[i + 1]["start_page"] - 1)
    if updated:
        updated[-1]["end_page"] = total_pages

    return updated, bool(milestone_msg), milestone_msg


def build_reading_memory_summary_fast(
    page_no: int,
    total_pages: int,
    active_chapter: str,
    extracted_questions: list[dict],
    detected_new_chapter: dict | None,
    pending_context: str,
    heading_stack: list[str],
) -> str:
    """快速規則生成工作記憶摘要。"""
    lines: list[str] = [
        f"【當前進度】：第 {page_no}/{total_pages} 頁 | 所在主章節：{active_chapter or '未定義'}",
    ]
    if heading_stack:
        lines.append(f"【標題層級棧】：{' -> '.join(heading_stack[-3:])}")

    if extracted_questions:
        sample_ids = [q["question_id"] for q in extracted_questions[:4]]
        if len(extracted_questions) > 4:
            sample_ids.append(f"...等共{len(extracted_questions)}題")
        lines.append(f"【本頁萃取題目】：{', '.join(sample_ids)}")
    else:
        lines.append("【本頁萃取題目】：無獨立題號（純描述或表頭）")

    if detected_new_chapter and detected_new_chapter.get("id"):
        cid = str(detected_new_chapter.get('id', '')).strip()
        ctitle = str(detected_new_chapter.get('title', '')).strip()
        name_str = f"{cid} {ctitle}".strip() if ctitle else cid
        lines.append(f"【章節轉折】：發現新章節 [{name_str}] 開始於本頁！")

    if pending_context:
        snippet = pending_context.strip().replace("\n", " ")[:60]
        lines.append(f"【未完結碎片】：頁底未閉合：「{snippet}...」")

    return "\n".join(lines)


async def generate_reading_memory_summary(
    prev_summary: str,
    page_no: int,
    total_pages: int,
    active_chapter: str,
    extracted_questions: list[dict],
    detected_new_chapter: dict | None,
    pending_context: str,
    heading_stack: list[str],
) -> str:
    """透過 LLM 生成自然語言工作記憶摘要。"""
    prompt = (
        f"當前進度：第 {page_no}/{total_pages} 頁，所在章節：{active_chapter}\n"
        f"標題層級棧：{' > '.join(heading_stack)}\n"
        f"本頁萃取題目數：{len(extracted_questions)}\n"
        f"章節轉折：{detected_new_chapter}\n"
        f"未完結碎片：{pending_context}\n"
        f"前一版摘要：{prev_summary}"
    )
    try:
        res = await chat_completion(
            prompt=prompt,
            system_prompt=get_langgraph_prompt("memory_summary"),
            temperature=0,
            max_tokens=512,
        )
        return res.strip()
    except Exception as e:
        logger.warning("[LangGraph Memory警告] 閱讀記憶摘要生成失敗：%s", e)
        return build_reading_memory_summary_fast(
            page_no=page_no,
            total_pages=total_pages,
            active_chapter=active_chapter,
            extracted_questions=extracted_questions,
            detected_new_chapter=detected_new_chapter,
            pending_context=pending_context,
            heading_stack=heading_stack,
        )
