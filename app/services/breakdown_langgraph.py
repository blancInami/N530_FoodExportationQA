"""
Breakdown LangGraph: 模擬人類循序閱讀與主動回讀問卷解析服務。

核心架構（Human-like Reading Flow）：
  1. Node 1 (TOC Extractor): 目錄與大綱結構分析，建立全局章節座標 (toc_structure)。
  2. Node 2 (Page Analyzer): 單頁循序研讀，優先縫合前頁 pending_context，並決定是否需回讀。
  3. Node 3 (Conditional Router): 判斷是否需要向後翻查 (Back-read)、繼續讀下一頁或結束。
  4. Node 4 (Back-Read Inspector): 從 Checkpoint 歷史中調取指定前頁影像與解析記錄，提供雙頁對照。
  5. Node 5 (Final Reducer): 全局統整、跨頁去重、題號標準化與偽題目過濾。
"""
import asyncio
import json
import logging
import re
import time
from typing import Annotated, Any, TypedDict
from dataclasses import dataclass, asdict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import get_settings
from app.lo.file_utils import expand_to_image_pages
from app.services.llm import chat_completion_vision, chat_completion

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Prompts: 目錄分析、單頁研讀、回讀對照與全局審查
# ═══════════════════════════════════════════════════════════════════════════

LANGGRAPH_TOC_SYSTEM_PROMPT = """\
你是一個精準的文件大綱結構分析專家。
你的任務是分析輸入的問卷前置頁面（通常包含目錄或主要章節），梳理整份問卷的「章節大綱樹（TOC Structure）」。

【執行規則】
1. 提取所有主章節（如 Part A, Chapter I, 1. General, 2.2 Standards 等）。
2. 為每個章節標註標準編號（如 "I", "II", "Part_A", "1"）與起始頁碼。

【輸出約束】
- 輸出合法的 JSON Array of Objects：
  [
    {
      "id": "章節標準編號 (如 I, II, Part_A, 1)",
      "title": "章節標題 (英文)",
      "page": 1
    }
  ]
- 絕對禁止包含任何解釋文字或 Markdown code fence。\
"""

LANGGRAPH_PAGE_ANALYZER_SYSTEM_PROMPT = """\
你是一個模擬人類專家循序審閱問卷的視覺多模態解析器（Human-like Page Inspector）。
你正在逐頁閱讀問卷影像。你具有以下職責：

【職責 1：前文未完結碎片縫合 (Pending Context Stitching)】
- 若提供了 <Pending_Context>，代表上一頁底部有一道被頁面切斷、尚未完結的題目。
- 請優先將本頁頂部的內容與 <Pending_Context> 拼接為完整題目！

【職責 2：實質問卷題目萃取】
- 僅萃取「需要受查國填答/說明」之實質題目（回答 Yes/No、提供數據、說明制度）。
- 嚴格剔除非題目雜訊：表格欄位標題（"No.", "Question", "Answer"）、填寫說明、法規純引文、純章節大標題。
- 題號必須扁平化展開（如 "II.1.a.i"），保留完整層級。

【職責 3：主動回讀判斷 (Back-reading Decision)】
- 若發現本頁頂部題目出現異常跳號（例如前頁題目是 1.1，本頁開頭卻是 "(c)" 且缺乏父級題號），且當前資訊不足以判定正確題號時：
  請在 "request_back_read" 欄位設定為 true，並指定 "back_read_pages": [前一頁頁碼]。
- 若本頁可正常完整解析，請將 "request_back_read" 設為 false。

【職責 4：本頁未完結碎片暫存】
- 若本頁最底部的題目在頁尾被截斷（例如只有前半句題幹或表格列未完），請將該碎片寫入 "pending_context" 欄位，交由下一頁縫合。

【輸出約束】
- 輸出合法的 JSON Object：
  {
    "extracted_questions": [
      {
        "question_id": "完整扁平化題號 (字串，如 1.1 或 II.1.a)",
        "question_text": "純英文題目內容 (不含題號前綴)"
      }
    ],
    "pending_context": "若頁底有未完結題幹則填寫，無則留空字串 \"\"",
    "active_chapter": "本頁結束時所處的主章節 (如 Chapter II)",
    "request_back_read": false,
    "back_read_pages": []
  }
- 絕對禁止在 JSON 前後加上任何解釋性文字或 Markdown 標籤。\
"""

LANGGRAPH_BACK_READ_SYSTEM_PROMPT = """\
你是記憶回讀專家（Back-Reading Inspector）。
你收到兩張影像：【前一頁影像（上一頁）】與【當前頁影像】以及先前的解析歷史。
你的任務是對照兩頁交界處的版面與文字，解決斷裂問題，並產出當前頁修正後的完整題目清單。

【輸出約束】
- 輸出合法的 JSON Object（同 Page Analyzer 格式）：
  {
    "extracted_questions": [
      {
        "question_id": "完整扁平化題號 (字串)",
        "question_text": "純英文題目內容"
      }
    ],
    "pending_context": "若頁底有未完結題幹則填寫，無則留空字串 \"\"",
    "active_chapter": "本頁結束時所處的主章節",
    "request_back_read": false,
    "back_read_pages": []
  }
- 不得包含 Markdown 標籤。\
"""

LANGGRAPH_REDUCER_SYSTEM_PROMPT = """\
你是問卷資料品質總審查員。你收到整份問卷循序逐頁研讀並縫合後的題目清單。
你的任務是進行最終的全局校驗與修整：
1. 刪除任何誤抓的表格表頭、填寫指南、簽名欄等殘留雜訊。
2. 跨頁重複題號若有漏未縫合之處，合併其 question_text。
3. 確保整體題號格式規範（點分十進位）。

【輸出】
修正與清洗後的 JSON Array of Objects，每個包含 "question_id" 與 "question_text"。
不得包含任何 Markdown code fence 或解釋文字。\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Graph State 定義
# ═══════════════════════════════════════════════════════════════════════════

class PageAnalysisResult(TypedDict, total=False):
    page_number: int
    extracted_questions: list[dict]
    pending_context: str
    active_chapter: str
    request_back_read: bool
    back_read_pages: list[int]


class QuestionnaireState(TypedDict, total=False):
    # 靜態文件資源
    image_pages: list[dict[str, str]]
    total_pages: int
    filename: str

    # 閱讀進度與大綱
    current_index: int
    toc_structure: list[dict]
    active_chapter: str
    pending_context: str

    # 累積結果與 Checkpoint 快照
    page_history: dict[int, PageAnalysisResult]
    final_questions: list[dict]

    # 回讀與循環防護
    back_read_context: str
    is_back_reading: bool
    back_read_count_for_page: int
    loop_count: int


# ═══════════════════════════════════════════════════════════════════════════
#  輔助函式 (JSON 清洗與驗證)
# ═══════════════════════════════════════════════════════════════════════════

def _clean_json(raw: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned.strip()


def _validate_extracted_questions(data: object) -> list[dict]:
    if not isinstance(data, list):
        return []
    validated: list[dict] = []
    for item in data:
        if isinstance(item, dict) and "question_id" in item and "question_text" in item:
            qid = str(item["question_id"]).strip()
            qtext = str(item["question_text"]).strip()
            if qid and qtext:
                validated.append({"question_id": qid, "question_text": qtext})
    return validated


def _deduplicate_items(items: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    result: list[dict] = []
    for item in items:
        qid = item["question_id"]
        if qid in seen:
            existing = seen[qid]
            normalized_existing = re.sub(r"\s+", " ", existing["question_text"]).strip()
            normalized_new = re.sub(r"\s+", " ", item["question_text"]).strip()
            if normalized_new and normalized_new not in normalized_existing:
                existing["question_text"] = f"{existing['question_text']} {item['question_text']}".strip()
        else:
            entry = {"question_id": qid, "question_text": item["question_text"]}
            seen[qid] = entry
            result.append(entry)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  LangGraph 節點實作 (Nodes)
# ═══════════════════════════════════════════════════════════════════════════

async def toc_extractor_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 1: 目錄大綱分析節點。"""
    pages = state["image_pages"]
    total = len(pages)
    if total <= 1:
        return {
            "toc_structure": [],
            "current_index": 0,
            "active_chapter": "",
            "pending_context": "",
            "page_history": {},
            "final_questions": [],
            "loop_count": 0,
        }

    sample_pages = pages[:min(3, total)]
    logger.info("[LangGraph TOC] 開始分析前 %d 頁目錄與大綱結構...", len(sample_pages))
    prompt = (
        f"這是一份共 {total} 頁問卷的前 {len(sample_pages)} 頁。"
        "請梳理出主章節架構，輸出 TOC 大綱 JSON。\n"
    )

    try:
        raw_output = await chat_completion_vision(
            prompt=prompt,
            images=sample_pages,
            system_prompt=LANGGRAPH_TOC_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=2048,
        )
        cleaned = _clean_json(raw_output)
        toc = json.loads(cleaned)
        if not isinstance(toc, list):
            toc = []
        logger.info("[LangGraph TOC完成] 成功建立目錄大綱，共 %d 個章節", len(toc))
    except Exception as e:
        logger.warning("[LangGraph TOC跳過] 目錄大綱分析失敗：%s", e)
        toc = []

    return {
        "toc_structure": toc,
        "current_index": 0,
        "active_chapter": toc[0]["id"] if toc else "",
        "pending_context": "",
        "page_history": {},
        "final_questions": [],
        "loop_count": 0,
        "is_back_reading": False,
        "back_read_count_for_page": 0,
    }


async def page_analyzer_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 2: 單頁循序研讀節點（支援前文縫合與主動回讀判定）。"""
    idx = state["current_index"]
    page_no = idx + 1
    total = state["total_pages"]
    page_img = state["image_pages"][idx]
    pending = state.get("pending_context", "")
    active_chap = state.get("active_chapter", "")
    toc = state.get("toc_structure", [])

    # 檢查 TOC 中是否有新章節在當前頁開始
    for item in toc:
        if isinstance(item, dict) and item.get("page") == page_no:
            active_chap = item.get("id", active_chap)

    logger.info(
        "[LangGraph Page] 研讀第 %d/%d 頁  當前主章節=%s  是否有前文碎片=%s",
        page_no, total, active_chap or "(無)", bool(pending),
    )
    t0 = time.perf_counter()

    prompt_parts = [
        f"這是問卷的第 {page_no}/{total} 頁影像。",
    ]
    if active_chap:
        prompt_parts.append(f"【當前所處章節】：{active_chap}（此頁題目請以該章節為前綴展開，如 {active_chap}.1）")
    if pending:
        prompt_parts.append(f"<Pending_Context>\n前一頁底部遺留未完結題幹：\n{pending}\n請優先將其與本頁開頭縫合！\n</Pending_Context>")

    prompt = "\n".join(prompt_parts)

    try:
        raw_output = await chat_completion_vision(
            prompt=prompt,
            images=[page_img],
            system_prompt=LANGGRAPH_PAGE_ANALYZER_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=4096,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        cleaned = _clean_json(raw_output)
        res = json.loads(cleaned)
        if not isinstance(res, dict):
            res = {}
    except Exception as e:
        logger.warning("[LangGraph Page警告] 第 %d 頁解析失敗：%s", page_no, e)
        res = {}

    extracted = _validate_extracted_questions(res.get("extracted_questions", []))
    new_pending = str(res.get("pending_context", "")).strip()
    new_active_chap = str(res.get("active_chapter", active_chap)).strip() or active_chap
    request_back_read = bool(res.get("request_back_read", False))
    back_read_pages = res.get("back_read_pages", [])

    logger.info(
        "[LangGraph Page完成] 第 %d/%d 頁  萃取題目=%d 筆  新碎片=%s  請求回讀=%s  耗時=%.1f ms",
        page_no, total, len(extracted), bool(new_pending), request_back_read, (time.perf_counter() - t0) * 1000,
    )

    # 記錄至歷史快照
    history = dict(state.get("page_history", {}))
    history[page_no] = PageAnalysisResult(
        page_number=page_no,
        extracted_questions=extracted,
        pending_context=new_pending,
        active_chapter=new_active_chap,
        request_back_read=request_back_read,
        back_read_pages=back_read_pages if isinstance(back_read_pages, list) else [],
    )

    return {
        "pending_context": new_pending,
        "active_chapter": new_active_chap,
        "page_history": history,
        "is_back_reading": request_back_read,
        "loop_count": state.get("loop_count", 0) + 1,
    }


async def back_read_inspector_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 4: 記憶回讀審查節點（提取前頁影像進行雙頁對照）。"""
    idx = state["current_index"]
    page_no = idx + 1
    total = state["total_pages"]
    prev_idx = max(0, idx - 1)
    prev_no = prev_idx + 1

    logger.info("[LangGraph Back-Read] 啟動記憶回讀審查：調取第 %d 頁與第 %d 頁雙影像對照...", prev_no, page_no)
    t0 = time.perf_counter()

    prev_img = state["image_pages"][prev_idx]
    curr_img = state["image_pages"][idx]
    prev_record = state.get("page_history", {}).get(prev_no, {})

    prompt = (
        f"這是問卷第 {prev_no} 頁（上一頁）與第 {page_no} 頁（當前頁）的連續影像。\n"
        f"上一頁解析結果概要：{json.dumps(prev_record.get('extracted_questions', []), ensure_ascii=False)}\n"
        f"請對照兩頁交界處，精確還原第 {page_no} 頁的所有正確題號與題目內容。\n"
    )

    try:
        raw_output = await chat_completion_vision(
            prompt=prompt,
            images=[prev_img, curr_img],
            system_prompt=LANGGRAPH_BACK_READ_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=4096,
        )
        cleaned = _clean_json(raw_output)
        res = json.loads(cleaned)
        if not isinstance(res, dict):
            res = {}
    except Exception as e:
        logger.warning("[LangGraph Back-Read警告] 回讀對照失敗：%s", e)
        res = {}

    extracted = _validate_extracted_questions(res.get("extracted_questions", []))
    new_pending = str(res.get("pending_context", "")).strip()

    # 更新當前頁快照
    history = dict(state.get("page_history", {}))
    history[page_no] = PageAnalysisResult(
        page_number=page_no,
        extracted_questions=extracted,
        pending_context=new_pending,
        active_chapter=state.get("active_chapter", ""),
        request_back_read=False,
        back_read_pages=[],
    )

    logger.info(
        "[LangGraph Back-Read完成] 雙頁對照修正完成：第 %d 頁修正題目=%d 筆  耗時=%.1f ms",
        page_no, len(extracted), (time.perf_counter() - t0) * 1000,
    )

    return {
        "page_history": history,
        "pending_context": new_pending,
        "is_back_reading": False,
        "back_read_count_for_page": state.get("back_read_count_for_page", 0) + 1,
    }


def advance_page_node(state: QuestionnaireState) -> QuestionnaireState:
    """推進至下一頁。"""
    return {
        "current_index": state["current_index"] + 1,
        "is_back_reading": False,
        "back_read_count_for_page": 0,
    }


async def final_reducer_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 5: 全局統整與去重節點。"""
    history = state.get("page_history", {})
    all_extracted: list[dict] = []
    for p in sorted(history.keys()):
        all_extracted.extend(history[p].get("extracted_questions", []))

    logger.info("[LangGraph Reducer] 匯總各頁題目，初始總筆數=%d 筆...", len(all_extracted))
    before_count = len(all_extracted)
    deduped = _deduplicate_items(all_extracted)
    logger.info("[LangGraph Reducer] 跨頁去重完成：%d 筆 -> %d 筆", before_count, len(deduped))

    # 全局 LLM 審查與清洗
    if deduped and len(deduped) <= 150:
        logger.info("[LangGraph Reducer] 啟動全局 LLM 格式與偽題目清洗 (題目數=%d 筆)...", len(deduped))
        t0 = time.perf_counter()
        try:
            prompt = json.dumps(deduped, ensure_ascii=False, indent=2)
            raw = await chat_completion(
                prompt=prompt,
                system_prompt=LANGGRAPH_REDUCER_SYSTEM_PROMPT,
                temperature=0,
                max_tokens=8192,
            )
            verified = json.loads(_clean_json(raw))
            valid = _validate_extracted_questions(verified)
            if valid:
                logger.info("[LangGraph Reducer完成] 全局清洗完成：%d 筆 -> %d 筆", len(deduped), len(valid))
                deduped = valid
        except Exception as e:
            logger.warning("[LangGraph Reducer跳過] 全局審查失敗：%s", e)
        logger.info("[LangGraph Reducer耗時] %.1f ms", (time.perf_counter() - t0) * 1000)

    return {"final_questions": deduped}


# ═══════════════════════════════════════════════════════════════════════════
#  條件路由 (Conditional Routing Logic)
# ═══════════════════════════════════════════════════════════════════════════

def page_routing_decision(state: QuestionnaireState) -> str:
    """
    決定單頁研讀後的下一步：
      - 'back_read': 觸發回讀審查
      - 'advance': 翻下一頁繼續研讀
      - 'finalize': 全部讀完，進入全局統整
    """
    is_back = state.get("is_back_reading", False)
    back_count = state.get("back_read_count_for_page", 0)
    current_idx = state.get("current_index", 0)
    total = state.get("total_pages", 1)

    # 若請求回讀且尚未超過單頁回讀上限 (最多 1 次)
    if is_back and back_count < 1 and current_idx > 0:
        return "back_read"

    # 若還有下一頁
    if current_idx < total - 1:
        return "advance"

    # 讀完最後一頁
    return "finalize"


# ═══════════════════════════════════════════════════════════════════════════
#  建立與編譯 LangGraph 工作流
# ═══════════════════════════════════════════════════════════════════════════

def build_questionnaire_graph() -> Any:
    """建構模擬人類循序研讀的 LangGraph 狀態機。"""
    workflow = StateGraph(QuestionnaireState)

    # 註冊節點
    workflow.add_node("toc_extractor", toc_extractor_node)
    workflow.add_node("page_analyzer", page_analyzer_node)
    workflow.add_node("back_read_inspector", back_read_inspector_node)
    workflow.add_node("advance_page", advance_page_node)
    workflow.add_node("final_reducer", final_reducer_node)

    # 建立邊與流向
    workflow.add_edge(START, "toc_extractor")
    workflow.add_edge("toc_extractor", "page_analyzer")

    workflow.add_conditional_edges(
        "page_analyzer",
        page_routing_decision,
        {
            "back_read": "back_read_inspector",
            "advance": "advance_page",
            "finalize": "final_reducer",
        },
    )

    # 回讀審查完成後，繼續決定是翻頁還是結束
    workflow.add_conditional_edges(
        "back_read_inspector",
        page_routing_decision,
        {
            "advance": "advance_page",
            "finalize": "final_reducer",
            "back_read": "advance_page", # 防禦回退
        },
    )

    workflow.add_edge("advance_page", "page_analyzer")
    workflow.add_edge("final_reducer", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ═══════════════════════════════════════════════════════════════════════════
#  主公開介面
# ═══════════════════════════════════════════════════════════════════════════

async def extract_questions_langgraph(file_bytes: bytes, filename: str) -> list[dict]:
    """
    使用 LangGraph 擬人化循序閱讀與主動回讀狀態機解析問卷。
    """
    pipeline_start = time.perf_counter()
    mime = "application/pdf" if filename.lower().endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    logger.info("[LangGraph 啟動] 檔名=%s  大小=%.2f KB", filename, len(file_bytes) / 1024)

    # 1. 沿用 LibreOffice + Poppler 展開為 JPEG
    pages = await expand_to_image_pages(
        raw=file_bytes,
        mime_type=mime,
        filename=filename,
    )
    if not pages:
        raise ValueError(f"無法將檔案 {filename} 轉換為有效頁面影像")

    total_pages = len(pages)
    logger.info("[LangGraph 展開完成] 共 %d 頁影像", total_pages)

    # 2. 初始化 State 與 Graph
    initial_state: QuestionnaireState = {
        "image_pages": pages,
        "total_pages": total_pages,
        "filename": filename,
        "current_index": 0,
        "toc_structure": [],
        "active_chapter": "",
        "pending_context": "",
        "page_history": {},
        "final_questions": [],
        "is_back_reading": False,
        "back_read_count_for_page": 0,
        "loop_count": 0,
    }

    graph = build_questionnaire_graph()
    config = {"configurable": {"thread_id": f"breakdown_{int(time.time()*1000)}"}}

    # 3. 執行狀態機
    final_output = await graph.ainvoke(initial_state, config=config)
    results = final_output.get("final_questions", [])

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info("[LangGraph 結束] 最終萃取題目數=%d 筆  總耗時=%.1f ms", len(results), total_ms)
    return results
