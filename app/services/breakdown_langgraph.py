"""
LangGraph Sequential Questionnaire Breakdown Engine (Human-like Page Inspector)
================================================================================
本模組提供基於 LangGraph 狀態機的問卷擬人化題目萃取服務。
包含 TOC 大綱解析、單頁視覺研讀、跨頁回讀對照與全局審查聚合。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.lo.file_utils import expand_to_image_pages
from app.schemas.langgraph import (
    OutlineItem,
    QuestionnaireState,
    TocAnchor,
)
from app.services.llm import chat_completion, chat_completion_vision
from app.utils.prompt_loader import get_langgraph_prompt
from app.utils.question_sanitizer import (
    canonicalize_all_question_ids,
    clean_json,
    detect_and_convert_weak_subsections,
    filter_false_positive_items,
    repair_nested_subquestion_hierarchy,
    repair_truncated_json_array,
    validate_extracted_questions,
)
from app.utils.toc_navigator import (
    build_page_section_map,
    build_reading_memory_summary_fast,
    calibrate_toc_anchors,
    calculate_toc_anchors,
    extract_section_prefix_from_qid,
    get_active_anchor,
    match_section_name,
    reorder_sections_by_physical_page,
    resolve_question_section,
    sanitize_depiction,
    split_section_id_and_depiction,
)

logger = logging.getLogger(__name__)

# ── 系統提示詞載入 (System Prompts) ──
LANGGRAPH_TOC_SYSTEM_PROMPT = get_langgraph_prompt("toc_indexer")
LANGGRAPH_PAGE_ANALYZER_SYSTEM_PROMPT = get_langgraph_prompt("page_analyzer")
LANGGRAPH_BACK_READ_SYSTEM_PROMPT = get_langgraph_prompt("back_read_inspector")
LANGGRAPH_REDUCER_SYSTEM_PROMPT = get_langgraph_prompt("reducer_quality")


# ═══════════════════════════════════════════════════════════════════════════
#  Graph Nodes (狀態機節點)
# ═══════════════════════════════════════════════════════════════════════════

async def toc_indexer_node(state: QuestionnaireState) -> QuestionnaireState:
    """
    Node 1: 自適應階層大綱預先索引節點 (TOC Pre-Indexer)。
    支援目錄優先解析，建立 page_section_map。
    """
    pages = state["image_pages"]
    total = state["total_pages"]
    t0 = time.perf_counter()

    if total <= 1:
        single_map, tree, s_order, s_dep = build_page_section_map([], total)
        return {
            "toc_anchors": [],
            "page_section_map": single_map,
            "global_outline_tree": tree,
            "current_index": 0,
            "active_chapter": s_order[0] if s_order else "",
            "heading_stack": [s_order[0]] if s_order else [],
            "pending_context": "",
            "reading_memory_summary": "",
            "page_history": {},
            "chapter_milestones": [],
            "final_questions": [],
            "loop_count": 0,
            "is_back_reading": False,
            "back_read_count_for_page": 0,
            "requested_back_read_pages": [],
            "back_read_trail": [],
            "section_depictions": s_dep,
            "section_order": s_order,
        }

    raw_outline_items: list[dict] = []

    # 僅從前 1~3 頁進行前置目錄解析
    sample_pages = pages[:min(3, total)]
    try:
        raw_output = await chat_completion_vision(
            prompt="請檢視此問卷前置頁面，識別並輸出整份問卷的階層章節大綱樹 (Hierarchical TOC Structure)。若前置頁無目錄頁則盡可能提取所見之章節標題。",
            images=sample_pages,
            system_prompt=LANGGRAPH_TOC_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=2048,
        )
        cleaned = clean_json(raw_output)
        toc = repair_truncated_json_array(cleaned)
        if isinstance(toc, list) and toc:
            raw_outline_items = [t for t in toc if isinstance(t, dict)]
            logger.info("[LangGraph TOC Indexer] 前置 1~3 頁目錄解析完成 (%d 個大綱節點)", len(raw_outline_items))
    except Exception as e:
        logger.warning("[LangGraph TOC Indexer] 目錄頁預析異常：%s", e)

    page_map, global_tree, section_order, section_depictions = build_page_section_map(raw_outline_items, total)
    anchors = calculate_toc_anchors(raw_outline_items, total)
    initial_chap = section_order[0] if section_order else (anchors[0]["id"] if anchors else "")

    logger.info(
        "[LangGraph TOC Indexer完成] 成功建立全篇階層地圖：主章節=%s  總大綱節點=%d  耗時=%.1f ms",
        ", ".join(section_order), len(global_tree), (time.perf_counter() - t0) * 1000,
    )

    return {
        "toc_anchors": anchors,
        "page_section_map": page_map,
        "global_outline_tree": global_tree,
        "current_index": 0,
        "active_chapter": initial_chap,
        "heading_stack": [initial_chap] if initial_chap else [],
        "pending_context": "",
        "reading_memory_summary": "",
        "page_history": {},
        "chapter_milestones": [],
        "final_questions": [],
        "loop_count": 0,
        "is_back_reading": False,
        "back_read_count_for_page": 0,
        "requested_back_read_pages": [],
        "back_read_trail": [],
        "section_depictions": section_depictions,
        "section_order": section_order,
    }


async def page_analyzer_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 2: 單頁循序研讀、結合預先索引地圖與工作記憶維護節點。"""
    idx = state["current_index"]
    page_no = idx + 1
    total = state["total_pages"]
    page_img = state["image_pages"][idx]
    pending = state.get("pending_context", "")
    reading_memory = state.get("reading_memory_summary", "")
    anchors = state.get("toc_anchors", [])
    active_anchor = get_active_anchor(anchors, page_no)
    prev_active_chap = active_anchor["id"] if active_anchor else state.get("active_chapter", "")
    heading_stack = state.get("heading_stack", [])
    milestones = list(state.get("chapter_milestones", []))
    page_section_map = state.get("page_section_map", {})
    pre_indexed_sections = page_section_map.get(page_no, [])

    t0 = time.perf_counter()

    prompt_parts: list[str] = [
        f"你正在研讀【第 {page_no}/{total} 頁】影像。",
    ]

    # 注入本頁預先索引的章節/小節地圖 (Map-Guided Navigation)
    if pre_indexed_sections:
        map_lines: list[str] = [f"<Page_Pre_Indexed_Sections>\n【本頁預先標記包含以下 {len(pre_indexed_sections)} 個章節/小節】："]
        for k, sec in enumerate(pre_indexed_sections):
            pos_label = sec.get("position", "top")
            sid = sec.get("section_id", "")
            sub = sec.get("subsection_id", sid)
            title = sec.get("title", "")
            is_maj = sec.get("is_major_section", False)
            dep = sec.get("depiction", "")
            flag = " <- 【新主章節起始】" if is_maj else ""
            map_lines.append(f"{k+1}. [位置 {pos_label}]: 主章節 [{sid}] / 小節 [{sub}] ({title}){flag}")
            if dep:
                map_lines.append(f'   前言說明: "{dep}"')
        if len(pre_indexed_sections) > 1:
            map_lines.append("【同頁切分指示】：請根據上述垂直位置，將上方題目標註為所屬舊區塊代號，下方題目標註為新區塊代號。")
        map_lines.append("</Page_Pre_Indexed_Sections>")
        prompt_parts.append("\n".join(map_lines))
    else:
        prompt_parts.append(
            f"<TOC_Navigation_Context>\n"
            f"全篇目錄導航預期本頁落於：[{active_anchor['id'] if active_anchor else '無預期'}]\n"
            f"目前繼承之層級標題棧：{' > '.join(heading_stack) if heading_stack else '頂層'}\n"
            f"</TOC_Navigation_Context>"
        )

    if reading_memory:
        prompt_parts.append(f"<Reading_Progress_Memory>\n前頁研讀累積摘要：\n{reading_memory}\n</Reading_Progress_Memory>")

    if pending:
        prompt_parts.append(
            f"<Pending_Context>\n前一頁底部遺留未完結題幹/表格碎片：\n{pending}\n【極重要銜接約束】：此碎片【只能且必須】與【本頁最頂部（第一行）開頭】無縫拼接並閉合！\n若本頁最頂部已是具備完整題號之全新題目，代表上一頁之內容已獨立完整，請直接解析本頁全新題目，嚴禁跨越本頁題目將前頁碎片與本頁中段或底部的其他表格錯誤拼接！\n</Pending_Context>"
        )

    prompt = "\n\n".join(prompt_parts)

    try:
        raw_output = await chat_completion_vision(
            prompt=prompt,
            images=[page_img],
            system_prompt=LANGGRAPH_PAGE_ANALYZER_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=4096,
        )
        cleaned = clean_json(raw_output)
        res = json.loads(cleaned)
        if not isinstance(res, dict):
            res = {}
    except Exception as e:
        logger.warning("[LangGraph Page警告] 第 %d 頁解析失敗：%s", page_no, e)
        res = {}

    raw_extracted = validate_extracted_questions(res.get("extracted_questions", []))
    filtered_extracted = filter_false_positive_items(raw_extracted)
    repaired_extracted = repair_nested_subquestion_hierarchy(filtered_extracted, default_parent_id=prev_active_chap)
    extracted, heading_stack = detect_and_convert_weak_subsections(repaired_extracted, heading_stack)
    new_pending = str(res.get("pending_context", "")).strip()
    detected_chap = res.get("detected_new_chapter")
    curr_heading = str(res.get("current_heading", "")).strip()
    request_back_read = bool(res.get("request_back_read", False))
    raw_back_pages = res.get("back_read_pages", [])

    calibrated_anchors = anchors
    section_depictions = dict(state.get("section_depictions", {}))
    section_order = list(state.get("section_order", []))

    new_clean_id = ""
    first_qid_of_new_chap = ""
    transition_idx = -1

    if detected_chap and isinstance(detected_chap, dict) and detected_chap.get("id"):
        calibrated_anchors, is_calibrated, m_msg = calibrate_toc_anchors(
            anchors=anchors,
            current_page=page_no,
            detected_chapter=detected_chap,
            total_pages=total,
        )
        if is_calibrated and m_msg:
            milestones.append(m_msg)
            heading_stack = [str(detected_chap["id"]).strip()]

        chap_id = str(detected_chap.get("id", "")).strip()
        chap_title = str(detected_chap.get("title", "")).strip()
        chap_depiction = str(detected_chap.get("depiction", "")).strip()
        first_qid_of_new_chap = str(detected_chap.get("first_question_id", "")).strip()

        clean_id, combined_dep = split_section_id_and_depiction(chap_id, chap_title, chap_depiction)
        if clean_id:
            new_clean_id = clean_id
            if clean_id not in section_order:
                section_depictions[clean_id] = combined_dep
                new_outline_item: OutlineItem = {
                    "section_id": clean_id,
                    "subsection_id": clean_id,
                    "title": chap_title or clean_id,
                    "page": page_no,
                    "position": "middle" if transition_idx != -1 else "top",
                    "is_major_section": True,
                    "depiction": combined_dep,
                }
                if page_no not in page_section_map:
                    page_section_map[page_no] = []
                page_section_map[page_no].append(new_outline_item)

                section_order = reorder_sections_by_physical_page(
                    outline_tree=list(state.get("global_outline_tree", [])),
                    page_section_map=page_section_map,
                    known_sections=section_order + [clean_id],
                )
            elif combined_dep and not section_depictions.get(clean_id):
                section_depictions[clean_id] = combined_dep
            elif combined_dep and combined_dep != section_depictions.get(clean_id, ""):
                if section_depictions.get(clean_id, "") not in combined_dep:
                    section_depictions[clean_id] = f"{section_depictions.get(clean_id, '')}\n\n{combined_dep}".strip()

    new_active_chap = new_clean_id or str(res.get("active_chapter", prev_active_chap)).strip() or prev_active_chap

    # 逐題精準歸屬
    transition_idx = -1
    if new_clean_id and new_clean_id != prev_active_chap:
        if first_qid_of_new_chap:
            for i, q in enumerate(extracted):
                if q["question_id"].lower() == first_qid_of_new_chap.lower():
                    transition_idx = i
                    break
        if transition_idx == -1:
            for i, q in enumerate(extracted):
                matched_sec = extract_section_prefix_from_qid(q["question_id"], section_order)
                if matched_sec == new_clean_id:
                    transition_idx = i
                    break
                if q.get("section_id") and match_section_name(q["section_id"], section_order) == new_clean_id:
                    transition_idx = i
                    break

    for i, q in enumerate(extracted):
        is_after_transition = (transition_idx != -1 and i >= transition_idx) or (transition_idx == -1 and bool(new_clean_id))
        assigned_section = resolve_question_section(
            q=q,
            page_prev_section=prev_active_chap,
            page_new_section=new_active_chap,
            known_sections=section_order,
            is_after_transition=is_after_transition,
        )
        q["_section_name"] = assigned_section

    back_read_pages: list[int] = []
    if isinstance(raw_back_pages, list):
        for p in raw_back_pages:
            try:
                p_int = int(p)
                if 1 <= p_int < page_no:
                    back_read_pages.append(p_int)
            except (ValueError, TypeError):
                continue
    if request_back_read and not back_read_pages and page_no > 1:
        back_read_pages = [page_no - 1]

    new_stack = list(heading_stack)
    if new_active_chap and (not new_stack or new_stack[0] != new_active_chap):
        new_stack = [new_active_chap]
    if curr_heading and curr_heading not in new_stack:
        new_stack.append(curr_heading)

    new_memory = build_reading_memory_summary_fast(
        page_no=page_no,
        total_pages=total,
        active_chapter=new_active_chap,
        extracted_questions=extracted,
        detected_new_chapter=detected_chap if isinstance(detected_chap, dict) else None,
        pending_context=new_pending,
        heading_stack=new_stack,
    )

    logger.info(
        "[LangGraph Page完成] 第 %d/%d 頁  萃取題目=%d 筆  新碎片=%s  請求回讀=%s (目標頁=%s)  耗時=%.1f ms",
        page_no, total, len(extracted), bool(new_pending), request_back_read, back_read_pages, (time.perf_counter() - t0) * 1000,
    )

    history = dict(state.get("page_history", {}))
    history[page_no] = {
        "page_number": page_no,
        "detected_new_chapter": detected_chap if isinstance(detected_chap, dict) else None,
        "extracted_questions": extracted,
        "pending_context": new_pending,
        "active_chapter": new_active_chap,
        "current_heading": curr_heading,
        "request_back_read": request_back_read,
        "back_read_pages": back_read_pages,
    }

    return {
        "toc_anchors": calibrated_anchors,
        "pending_context": new_pending,
        "active_chapter": new_active_chap,
        "heading_stack": new_stack,
        "reading_memory_summary": new_memory,
        "page_history": history,
        "chapter_milestones": milestones,
        "requested_back_read_pages": back_read_pages,
        "is_back_reading": request_back_read and bool(back_read_pages),
        "loop_count": state.get("loop_count", 0) + 1,
        "section_depictions": section_depictions,
        "section_order": section_order,
    }


async def back_read_inspector_node(state: QuestionnaireState) -> QuestionnaireState:
    idx = state["current_index"]
    page_no = idx + 1
    total = state["total_pages"]
    target_pages = state.get("requested_back_read_pages", [])
    if not target_pages:
        target_pages = [page_no - 1] if page_no > 1 else []

    if not target_pages:
        return {
            "is_back_reading": False,
            "requested_back_read_pages": [],
            "back_read_count_for_page": state.get("back_read_count_for_page", 0) + 1,
        }

    history = state.get("page_history", {})
    all_pages = state["image_pages"]

    pages_to_send: list[dict] = []
    seen_p = set()
    for p in target_pages:
        if 1 <= p <= total and p not in seen_p:
            seen_p.add(p)
            pages_to_send.append(all_pages[p - 1])

    if page_no not in seen_p:
        pages_to_send.append(all_pages[idx])

    pages_str = ", ".join(f"p.{p}" for p in target_pages)
    trail_msg = f"第 {page_no} 頁向後回讀 [{pages_str}] 共同對照修復"
    logger.info("[LangGraph Back-Read啟動] %s (附帶 %d 頁影像)", trail_msg, len(pages_to_send))
    t0 = time.perf_counter()

    context_parts: list[str] = [
        f"【多頁回讀對照任務】：請同時審查前置關聯頁面與當前第 {page_no} 頁影像，解決題號斷層與遺漏問題。",
        f"回讀目標頁面：{pages_str}，當前頁碼：p.{page_no} (共 {total} 頁)。",
    ]
    for p in target_pages:
        if p in history:
            prev_res = history[p]
            q_ids = [q["question_id"] for q in prev_res.get("extracted_questions", [])]
            context_parts.append(
                f"<History_Page_{p}>\n"
                f"該頁章節：{prev_res.get('active_chapter', '')} | 題號清單：{q_ids}\n"
                f"頁底未完結碎片：{prev_res.get('pending_context', '')}\n"
                f"</History_Page_{p}>"
            )

    prompt = "\n\n".join(context_parts)
    try:
        raw_output = await chat_completion_vision(
            prompt=prompt,
            images=pages_to_send,
            system_prompt=LANGGRAPH_BACK_READ_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=4096,
        )
        cleaned = clean_json(raw_output)
        res = json.loads(cleaned)
        if not isinstance(res, dict):
            res = {}
    except Exception as e:
        logger.warning("[LangGraph Back-Read警告] 多頁回讀對照失敗：%s", e)
        res = {}

    prev_active_chap = state.get("active_chapter", "")
    heading_stack = state.get("heading_stack", [])

    raw_extracted = validate_extracted_questions(res.get("extracted_questions", []))
    filtered_extracted = filter_false_positive_items(raw_extracted)
    repaired_extracted = repair_nested_subquestion_hierarchy(filtered_extracted, default_parent_id=prev_active_chap)
    extracted, heading_stack = detect_and_convert_weak_subsections(repaired_extracted, heading_stack)
    new_pending = str(res.get("pending_context", "")).strip()

    section_order = list(state.get("section_order", []))
    active_chap = state.get("active_chapter", "")
    for q in extracted:
        assigned_section = resolve_question_section(
            q=q,
            page_prev_section=active_chap,
            page_new_section=active_chap,
            known_sections=section_order,
            is_after_transition=True,
        )
        q["_section_name"] = assigned_section

    hist_copy = dict(history)
    hist_copy[page_no] = {
        "page_number": page_no,
        "detected_new_chapter": None,
        "extracted_questions": extracted,
        "pending_context": new_pending,
        "active_chapter": active_chap,
        "current_heading": res.get("current_heading", ""),
        "request_back_read": False,
        "back_read_pages": [],
    }

    trails = list(state.get("back_read_trail", []))
    trails.append(trail_msg)

    logger.info(
        "[LangGraph Back-Read完成] 多頁對照修正完成：第 %d 頁修正題目=%d 筆  耗時=%.1f ms",
        page_no, len(extracted), (time.perf_counter() - t0) * 1000,
    )

    return {
        "page_history": hist_copy,
        "pending_context": new_pending,
        "is_back_reading": False,
        "requested_back_read_pages": [],
        "back_read_count_for_page": state.get("back_read_count_for_page", 0) + 1,
        "back_read_trail": trails,
    }


def advance_page_node(state: QuestionnaireState) -> QuestionnaireState:
    return {
        "current_index": state["current_index"] + 1,
        "is_back_reading": False,
        "requested_back_read_pages": [],
        "back_read_count_for_page": 0,
    }


async def final_reducer_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 5: 全域確定性題號前綴歸屬、實體頁碼拓撲排序、純文字平行審查與格式聚合節點。"""
    history = state.get("page_history", {})
    section_depictions = dict(state.get("section_depictions", {}))
    section_order = list(state.get("section_order", []))

    appearance_order: list[str] = []
    section_questions: dict[str, list[dict]] = {}

    for p in sorted(history.keys()):
        for q in history[p].get("extracted_questions", []):
            if isinstance(q, dict):
                raw_qid = q.get("question_id", "")
                sec = q.pop("_section_name", "")
                corrected_sec = extract_section_prefix_from_qid(raw_qid, section_order)
                final_sec = corrected_sec if corrected_sec else (sec or (section_order[0] if section_order else "General"))

                if final_sec not in section_questions:
                    section_questions[final_sec] = []
                    if final_sec and final_sec not in appearance_order:
                        appearance_order.append(final_sec)
                section_questions[final_sec].append(q)

    section_first_appearance: dict[str, tuple[int, int]] = {}
    for p in sorted(history.keys()):
        for idx_in_page, q in enumerate(history[p].get("extracted_questions", [])):
            raw_qid = q.get("question_id", "")
            sec = q.get("_section_name", "")
            corrected_sec = extract_section_prefix_from_qid(raw_qid, section_order)
            target_sec = corrected_sec if corrected_sec else (sec or (section_order[0] if section_order else "General"))
            if target_sec and target_sec not in section_first_appearance:
                section_first_appearance[target_sec] = (p, idx_in_page)

    all_candidate_secs = list(dict.fromkeys([sn for sn in appearance_order if sn] + [sn for sn in section_order if sn in section_questions]))
    ordered_sections = sorted(
        all_candidate_secs,
        key=lambda sn: section_first_appearance.get(sn, (9999, 9999))
    )

    if not ordered_sections:
        ordered_sections = ["General"]
        section_questions["General"] = []
    elif len(ordered_sections) == 1 and ordered_sections[0] == "":
        section_questions["General"] = section_questions.pop("")
        ordered_sections = ["General"]
    elif "" in section_questions:
        first_named = ordered_sections[0] if ordered_sections[0] else "General"
        if first_named in section_questions:
            section_questions[first_named] = section_questions.pop("") + section_questions[first_named]
        else:
            section_questions[first_named] = section_questions.pop("")
        if "" in ordered_sections:
            ordered_sections.remove("")

    total_raw = sum(len(qs) for qs in section_questions.values())
    logger.info("[LangGraph Reducer] 匯總各頁題目，初始總筆數=%d 筆，共 %d 個區塊...", total_raw, len(ordered_sections))

    settings = get_settings()
    max_concurrency = max(2, getattr(settings, "breakdown_max_concurrency", 4))
    sem = asyncio.Semaphore(max_concurrency)

    async def _verify_reducer_batch(batch_items: list[dict], section_name: str, depiction: str, b_idx: int) -> list[dict]:
        async with sem:
            try:
                reducer_input = [{
                    "section_name": section_name,
                    "depiction": depiction,
                    "questions": batch_items,
                }]
                prompt = json.dumps(reducer_input, ensure_ascii=False, indent=2)
                raw = await chat_completion(
                    prompt=prompt,
                    system_prompt=LANGGRAPH_REDUCER_SYSTEM_PROMPT,
                    temperature=0,
                    max_tokens=8192,
                )
                repaired = repair_truncated_json_array(raw)
                if repaired:
                    first = repaired[0]
                    if isinstance(first, dict) and "questions" in first:
                        questions_raw = first.get("questions", [])
                        valid = validate_extracted_questions(questions_raw)
                        if valid:
                            return valid
                    else:
                        valid = validate_extracted_questions(repaired)
                        if valid:
                            return valid
                return batch_items
            except Exception as e:
                logger.warning("[LangGraph Reducer警告] 區塊 '%s' 批次 %d 審查異常，保留原始項目：%s", section_name, b_idx + 1, e)
                return batch_items

    t0 = time.perf_counter()

    async def _verify_single_section(sec_name: str) -> tuple[str, list[dict]]:
        sec_items = section_questions.get(sec_name, [])
        depiction = section_depictions.get(sec_name, "")
        deduped = canonicalize_all_question_ids(sec_items)

        if not deduped:
            return sec_name, []

        batch_size = 80
        batches = [deduped[i:i + batch_size] for i in range(0, len(deduped), batch_size)]
        total_batches = len(batches)
        logger.info("[LangGraph Reducer] 區塊 '%s' 啟動平行文字語義審查（%d 批次，%d 筆）...", sec_name, total_batches, len(deduped))

        tasks = [_verify_reducer_batch(batches[i], sec_name, depiction, i) for i in range(total_batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        verified_items: list[dict] = []
        for i, res in enumerate(results):
            if isinstance(res, list):
                verified_items.extend(res)
            else:
                verified_items.extend(batches[i])

        return sec_name, canonicalize_all_question_ids(verified_items)

    sec_tasks = [_verify_single_section(sn) for sn in ordered_sections]
    sec_results = await asyncio.gather(*sec_tasks, return_exceptions=True)

    final_sections: list[dict] = []
    seen_sec_names: set[str] = set()
    for i, res in enumerate(sec_results):
        sec_name = ordered_sections[i]
        if sec_name in seen_sec_names:
            continue
        seen_sec_names.add(sec_name)

        final_canonical = res[1] if isinstance(res, tuple) else canonicalize_all_question_ids(section_questions.get(sec_name, []))
        depiction = section_depictions.get(sec_name, "")
        clean_sanitized_dep = sanitize_depiction(depiction, sec_name, final_canonical)

        final_sections.append({
            sec_name: {
                "depiction": clean_sanitized_dep,
                "question": [
                    {
                        "question_id": q["question_id"],
                        "question_text": q["question_text"],
                    }
                    for q in final_canonical
                ],
                "questions": [
                    {
                        "question_id": q["question_id"],
                        "question_text": q["question_text"],
                    }
                    for q in final_canonical
                ],
            }
        })
        logger.info("[LangGraph Reducer] 區塊 '%s' 完成：depiction=%s  題目=%d 筆", sec_name, bool(depiction), len(final_canonical))

    elapsed = (time.perf_counter() - t0) * 1000
    total_final = sum(len(b[list(b.keys())[0]]["question"]) for b in final_sections)
    logger.info("[LangGraph Reducer結束] 耗時=%.1f ms  最終區塊數=%d 個  總題目數=%d 筆", elapsed, len(final_sections), total_final)

    return {
        "final_questions": final_sections,
    }


def page_routing_decision(state: QuestionnaireState) -> str:
    current_idx = state["current_index"]
    total = state["total_pages"]
    back_read_count = state.get("back_read_count_for_page", 0)

    if state.get("is_back_reading") and back_read_count < 1:
        return "back_read"

    if current_idx < total - 1:
        return "advance"

    return "finalize"


def build_questionnaire_graph() -> Any:
    workflow = StateGraph(QuestionnaireState)

    workflow.add_node("toc_indexer", toc_indexer_node)
    workflow.add_node("page_analyzer", page_analyzer_node)
    workflow.add_node("back_read_inspector", back_read_inspector_node)
    workflow.add_node("advance_page", advance_page_node)
    workflow.add_node("final_reducer", final_reducer_node)

    workflow.add_edge(START, "toc_indexer")
    workflow.add_edge("toc_indexer", "page_analyzer")

    workflow.add_conditional_edges(
        "page_analyzer",
        page_routing_decision,
        {
            "back_read": "back_read_inspector",
            "advance": "advance_page",
            "finalize": "final_reducer",
        },
    )

    workflow.add_conditional_edges(
        "back_read_inspector",
        page_routing_decision,
        {
            "advance": "advance_page",
            "finalize": "final_reducer",
            "back_read": "advance_page",
        },
    )

    workflow.add_edge("advance_page", "page_analyzer")
    workflow.add_edge("final_reducer", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


async def extract_questions_langgraph(file_bytes: bytes, filename: str) -> list[dict]:
    pipeline_start = time.perf_counter()
    mime = "application/pdf" if filename.lower().endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    logger.info("[LangGraph 啟動] 檔名=%s  大小=%.2f KB", filename, len(file_bytes) / 1024)

    pages = await expand_to_image_pages(
        raw=file_bytes,
        mime_type=mime,
        filename=filename,
    )
    if not pages:
        raise ValueError(f"無法將檔案 {filename} 轉換為有效頁面影像")

    total_pages = len(pages)
    logger.info("[LangGraph 展開完成] 共 %d 頁影像", total_pages)

    initial_state: QuestionnaireState = {
        "image_pages": pages,
        "total_pages": total_pages,
        "filename": filename,
        "current_index": 0,
        "toc_anchors": [],
        "page_section_map": {},
        "global_outline_tree": [],
        "active_chapter": "",
        "heading_stack": [],
        "pending_context": "",
        "reading_memory_summary": "",
        "page_history": {},
        "chapter_milestones": [],
        "final_questions": [],
        "requested_back_read_pages": [],
        "is_back_reading": False,
        "back_read_count_for_page": 0,
        "back_read_trail": [],
        "loop_count": 0,
        "section_depictions": {},
        "section_order": [],
    }

    graph = build_questionnaire_graph()
    config = {"configurable": {"thread_id": f"breakdown_{int(time.time()*1000)}"}}

    final_output = await graph.ainvoke(initial_state, config=config)
    results = final_output.get("final_questions", [])

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    milestones = final_output.get("chapter_milestones", [])
    if milestones:
        logger.info("[LangGraph 關鍵章節校準里程碑] %s", " | ".join(milestones))
    trail = final_output.get("back_read_trail", [])
    if trail:
        logger.info("[LangGraph 回讀翻閱軌跡] %s", " | ".join(trail))

    total_q_count = 0
    if isinstance(results, list):
        for section_block in results:
            if isinstance(section_block, dict):
                for sec_detail in section_block.values():
                    if isinstance(sec_detail, dict):
                        total_q_count += len(sec_detail.get("questions", sec_detail.get("question", [])))
    logger.info("[LangGraph 結束] 最終萃取區塊數=%d  總題目數=%d 筆  總耗時=%.1f ms", len(results), total_q_count, total_ms)
    return results
