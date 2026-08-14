"""
Breakdown LangGraph: 模擬人類循序閱讀、動態章節校準、目錄區間錨定與任意指定頁記憶回讀問卷解析服務。

核心架構特性（Human-like Reading Flow 2.1）：
  1. TOC Expectation (Node 1: TOC Extractor):
     分析前置頁，梳理出整份問卷的章節大綱樹（預期章節名稱與相對順序）。
  2. Dynamic Page Calibration & Working Memory (Node 2: Page Analyzer):
     逐頁研讀時，一旦畫面出現【新章節大標題/關鍵轉折點】（detected_new_chapter），
     立即動態將該章節的真實物理起始頁錨定為當前頁面 (start_page = current_page)，
     前一章節 end_page 自動閉合為 (current_page - 1)，並刷新 active_chapter 與 heading_stack。
  3. Ground-Truth Arbitrary Back-Reading (Node 4: Back-Read Inspector):
     打破目錄自訂頁碼與物理頁碼偏差問題，當後續頁面需要回讀章節定義時，
     直接調取動態校準後的【真實物理關鍵起始頁 + 當前頁】影像對照還原。
  4. Safety Loop & Budget Guard (Node 3: Router):
     單頁最多回讀 1 次，限制回讀最多打包 3 頁影像，記錄翻頁軌跡並防止無限回讀死鎖。
  5. Final Reducer (Node 5):
     跨頁去重、格式規範化與偽題目二次過濾。
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
你的任務是分析輸入的問卷前置頁面（通常包含目錄或主要章節），梳理整份問卷的「章節大綱樹（TOC Structure）」與預期順序。

【執行規則】
1. 提取所有主章節（如 Part A, Chapter I, 1. General, 2.2 Standards 等）。
2. 為每個章節標註標準編號（如 "I", "II", "Part_A", "1"）與標題。

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
你正在逐頁閱讀問卷影像。你擁有目錄導航視野（TOC Anchors）與層級標題棧（Heading Stack）。

【職責 1：章節轉折與關鍵頁面偵測 (Key Chapter Transition & Memory Calibration)】
- 當你在本頁頂部或中段發現【新的主章節/大標題】開始（例如 "Chapter II. Animal Health", "Part B. Processing Standards" 等）：
  請在 "detected_new_chapter" 填入該章節資訊，系統將立即把本頁錨定為該章節的真實物理起始頁，以校準工作記憶！
- 若本頁只是延續既有章節，請將 "detected_new_chapter" 設為 null。

【職責 2：前文未完結碎片縫合 (Pending Context Stitching)】
- 若提供了 <Pending_Context>，代表上一頁底部有一道被頁面切斷、尚未完結的題目。
- 請優先將本頁頂部的內容與 <Pending_Context> 拼接為完整題目！

【職責 3：實質問卷題目萃取】
- 僅萃取「需要受查國填答/說明」之實質題目（回答 Yes/No、提供數據、說明制度）。
- 嚴格剔除非題目雜訊：表格欄位標題（"No.", "Question", "Answer"）、填寫說明、法規純引文、純章節大標題。
- 題號必須扁平化展開（如 "II.1.a.i"），完整繼承章節與父題號。

【職責 4：主動請求任意頁回讀 (Arbitrary Page Back-reading)】
- 若發現本頁頂部題目出現異常跳號（例如本頁開頭是 "(d)" 或 "2.3"，但缺乏父級定義），
  且你從 <TOC_Navigation_Context> 或記憶中知道其根章節落在第 X 頁（例如第 3 頁）：
  請將 "request_back_read" 設為 true，並在 "back_read_pages" 指定欲翻閱的頁碼列表（例如 [3] 或 [3, 7]）。
- 若本頁可正常完整解析，請將 "request_back_read" 設為 false，"back_read_pages" 設為 []。

【職責 5：本頁未完結碎片暫存】
- 若本頁最底部的題目在頁尾被截斷（例如只有前半句題幹或表格列未完），請將該碎片寫入 "pending_context" 欄位，交由下一頁縫合。

【輸出約束】
- 輸出合法的 JSON Object：
  {
    "detected_new_chapter": {
      "id": "新章節標準編號 (如 Chapter_II, Part_B)",
      "title": "新章節標題"
    } 或 null,
    "extracted_questions": [
      {
        "question_id": "完整扁平化題號 (字串，如 1.1 或 II.1.a)",
        "question_text": "純英文題目內容 (不含題號前綴)"
      }
    ],
    "pending_context": "若頁底有未完結題幹則填寫，無則留空字串 \"\"",
    "active_chapter": "本頁結束時所處的主章節 (如 Chapter II)",
    "current_heading": "本頁最後見到的子標題 (如 2.2.1 Water Quality)",
    "request_back_read": false,
    "back_read_pages": []
  }
- 絕對禁止在 JSON 前後加上任何解釋性文字或 Markdown 標籤。\
"""

LANGGRAPH_BACK_READ_SYSTEM_PROMPT = """\
你是記憶回讀專家（Back-Reading Inspector）。
你收到了多張跨頁影像（包含根章節定義頁、前一頁與當前頁）以及先前的解析歷史。
你的任務是對照這些關聯頁面，解決斷層與題號遺漏問題，精確還原【當前頁】的所有正確題號與題目內容。

【輸出約束】
- 輸出合法的 JSON Object（同 Page Analyzer 格式）：
  {
    "detected_new_chapter": null,
    "extracted_questions": [
      {
        "question_id": "完整扁平化題號 (字串)",
        "question_text": "純英文題目內容"
      }
    ],
    "pending_context": "若頁底有未完結題幹則填寫，無則留空字串 \"\"",
    "active_chapter": "本頁結束時所處的主章節",
    "current_heading": "本頁最後見到的子標題",
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

class TocAnchor(TypedDict, total=False):
    id: str
    title: str
    start_page: int
    end_page: int
    is_calibrated: bool  # 是否已經在物理頁面研讀中動態校準確認


class PageAnalysisResult(TypedDict, total=False):
    page_number: int
    detected_new_chapter: dict | None
    extracted_questions: list[dict]
    pending_context: str
    active_chapter: str
    current_heading: str
    request_back_read: bool
    back_read_pages: list[int]


class QuestionnaireState(TypedDict, total=False):
    # 靜態文件資源
    image_pages: list[dict[str, str]]
    total_pages: int
    filename: str

    # 閱讀進度與大綱錨定
    current_index: int
    toc_structure: list[dict]
    toc_anchors: list[TocAnchor]
    active_chapter: str
    heading_stack: list[str]
    pending_context: str

    # 累積結果與 Checkpoint 快照
    page_history: dict[int, PageAnalysisResult]
    chapter_milestones: list[str]
    final_questions: list[dict]

    # 回讀與循環防護
    requested_back_read_pages: list[int]
    is_back_reading: bool
    back_read_count_for_page: int
    back_read_trail: list[str]
    loop_count: int


# ═══════════════════════════════════════════════════════════════════════════
#  輔助函式 (JSON 清洗、驗證、TOC 區間計算與動態校準)
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


def _normalize_single_question_id(raw_id: str) -> str:
    """單題號基礎標點與符號清洗。"""
    qid = str(raw_id).strip()
    # 1. 將 Part_A, Part_1, Chapter_II 等底線轉為標準空格
    qid = re.sub(r"\b(Part|Chapter|Section|Annex|Appendix)_([A-Za-z0-9IVXLCDM]+)\b", r"\1 \2", qid, flags=re.IGNORECASE)
    # 2. 清除底線為點號或空格（若非標準關鍵字）
    qid = re.sub(r"_+", ".", qid)
    # 3. 合併多餘連續點號，例如 Part C..1 -> Part C.1
    qid = re.sub(r"\.+", ".", qid)
    # 4. 去除點號兩側多餘空格，例如 "Part C . 1" -> "Part C.1"
    qid = re.sub(r"\s*\.\s*", ".", qid)
    # 5. 去除首尾多餘點號或空格（但保留內部格式）
    qid = qid.strip(". ").strip()
    # 6. 修復 "Part A. 1" 或 "Part A 1" 模式為 "Part A.1"
    qid = re.sub(r"\b(Part\s+[A-Za-z0-9IVXLCDM]+)\s+(\d+)\b", r"\1.\2", qid, flags=re.IGNORECASE)
    return qid


def _canonicalize_all_question_ids(items: list[dict]) -> list[dict]:
    """
    全篇題號統一性清洗與前綴對齊：
    1. 執行基礎單題號清洗。
    2. 統計全篇主流前綴模式（例如若檢測到多個 'Part A', 'Part B' 等）：
       - 若後續出現單獨字母開頭（如 'D.1.a', 'E.1'），自動補齊為 'Part D.1.a'。
    3. 執行去重與格式保證。
    """
    if not items:
        return []

    # 第一階段：單項基礎清洗
    cleaned_items: list[dict] = []
    part_letter_seen: set[str] = set()

    for item in items:
        raw_id = item.get("question_id", "")
        raw_text = item.get("question_text", "")
        norm_id = _normalize_single_question_id(raw_id)
        if norm_id and raw_text:
            cleaned_items.append({"question_id": norm_id, "question_text": raw_text})
            # 檢查是否有 "Part X" 模式
            m = re.match(r"^Part\s+([A-Z])(?:\.|$)", norm_id, re.IGNORECASE)
            if m:
                part_letter_seen.add(m.group(1).upper())

    # 第二階段：若文檔主流存在 Part A/B/C... 模式，對齊遺漏了 "Part " 前綴的單字母題號
    if len(part_letter_seen) >= 2:
        for item in cleaned_items:
            qid = item["question_id"]
            # 匹配如 "D.1.a", "E.2" 等孤立大寫字母開頭且無 Part 前綴
            m_bare = re.match(r"^([A-Z])(\.\d+.*)$", qid)
            if m_bare:
                letter = m_bare.group(1).upper()
                rest = m_bare.group(2)
                # 補齊為 "Part D.1.a"
                item["question_id"] = f"Part {letter}{rest}"

    return _deduplicate_items(cleaned_items)


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


def _calculate_toc_anchors(toc: list[dict], total_pages: int) -> list[TocAnchor]:
    """根據 TOC 列表計算初始預估的章節區間。"""
    if not toc:
        return []

    valid_items = []
    for item in toc:
        if isinstance(item, dict) and "id" in item:
            try:
                page = int(item.get("page", 1))
            except (ValueError, TypeError):
                page = 1
            valid_items.append({
                "id": str(item["id"]).strip(),
                "title": str(item.get("title", "")).strip(),
                "page": max(1, min(page, total_pages)),
            })

    valid_items.sort(key=lambda x: x["page"])
    anchors: list[TocAnchor] = []
    for i, curr in enumerate(valid_items):
        start_p = curr["page"]
        if i + 1 < len(valid_items):
            end_p = max(start_p, valid_items[i + 1]["page"] - 1)
        else:
            end_p = total_pages
        anchors.append({
            "id": curr["id"],
            "title": curr["title"],
            "start_page": start_p,
            "end_page": end_p,
            "is_calibrated": False,
        })
    return anchors


def _get_active_anchor(anchors: list[TocAnchor], page_no: int) -> TocAnchor | None:
    """根據頁碼獲取當前所屬的 TOC 錨點。"""
    for a in anchors:
        if a["start_page"] <= page_no <= a["end_page"]:
            return a
    return anchors[0] if anchors else None


def _calibrate_toc_anchors(
    anchors: list[TocAnchor],
    current_page: int,
    detected_chapter: dict,
    total_pages: int,
) -> tuple[list[TocAnchor], bool, str]:
    """
    當在物理頁面 current_page 發現新章節時，即時動態校準 TOC 錨點區間。
    回傳 (校準後 anchors, 是否發生校準, 紀錄文字)。
    """
    if not detected_chapter or not isinstance(detected_chapter, dict):
        return anchors, False, ""

    new_id = str(detected_chapter.get("id", "")).strip()
    new_title = str(detected_chapter.get("title", "")).strip()
    if not new_id:
        return anchors, False, ""

    new_anchors = [dict(a) for a in anchors]
    matched_idx = -1

    # 尋找是否在既有 TOC 預期列表中（比對 ID 或 title 包含關係）
    for i, a in enumerate(new_anchors):
        if a["id"].lower() == new_id.lower() or (new_title and new_title.lower() in a.get("title", "").lower()):
            matched_idx = i
            break

    milestone_msg = ""
    if matched_idx >= 0:
        # 已有預期章節：校準其 start_page
        old_start = new_anchors[matched_idx]["start_page"]
        new_anchors[matched_idx]["start_page"] = current_page
        new_anchors[matched_idx]["is_calibrated"] = True
        if new_title and not new_anchors[matched_idx].get("title"):
            new_anchors[matched_idx]["title"] = new_title

        # 前一章節的 end_page 閉合至 current_page - 1
        if matched_idx > 0:
            new_anchors[matched_idx - 1]["end_page"] = max(new_anchors[matched_idx - 1]["start_page"], current_page - 1)

        milestone_msg = f"Chapter '{new_id}' calibrated from p.{old_start} -> real physical p.{current_page}"
        logger.info("[TOC動態校準] %s", milestone_msg)
    else:
        # 新增未在目錄出現的突發章節
        # 先閉合前一個錨點
        if new_anchors:
            new_anchors[-1]["end_page"] = max(new_anchors[-1]["start_page"], current_page - 1)
        new_anchor: TocAnchor = {
            "id": new_id,
            "title": new_title,
            "start_page": current_page,
            "end_page": total_pages,
            "is_calibrated": True,
        }
        new_anchors.append(new_anchor)
        milestone_msg = f"Discovered new chapter '{new_id}' anchored at real physical p.{current_page}"
        logger.info("[TOC動態發現] %s", milestone_msg)

    # 確保所有 anchors 頁碼單調遞增
    new_anchors.sort(key=lambda x: x["start_page"])
    for i in range(len(new_anchors) - 1):
        new_anchors[i]["end_page"] = max(new_anchors[i]["start_page"], new_anchors[i + 1]["start_page"] - 1)
    if new_anchors:
        new_anchors[-1]["end_page"] = total_pages

    return new_anchors, True, milestone_msg


# ═══════════════════════════════════════════════════════════════════════════
#  LangGraph 節點實作 (Nodes)
# ═══════════════════════════════════════════════════════════════════════════

async def toc_extractor_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 1: 目錄大綱與初始預估錨定節點。"""
    pages = state["image_pages"]
    total = len(pages)
    if total <= 1:
        return {
            "toc_structure": [],
            "toc_anchors": [],
            "current_index": 0,
            "active_chapter": "",
            "heading_stack": [],
            "pending_context": "",
            "page_history": {},
            "chapter_milestones": [],
            "final_questions": [],
            "loop_count": 0,
            "back_read_trail": [],
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
    except Exception as e:
        logger.warning("[LangGraph TOC跳過] 目錄大綱分析失敗：%s", e)
        toc = []

    anchors = _calculate_toc_anchors(toc, total)
    initial_chap = anchors[0]["id"] if anchors else ""
    logger.info("[LangGraph TOC完成] 成功建立目錄大綱，共 %d 個初始章節錨點", len(anchors))

    return {
        "toc_structure": toc,
        "toc_anchors": anchors,
        "current_index": 0,
        "active_chapter": initial_chap,
        "heading_stack": [initial_chap] if initial_chap else [],
        "pending_context": "",
        "page_history": {},
        "chapter_milestones": [],
        "final_questions": [],
        "loop_count": 0,
        "is_back_reading": False,
        "back_read_count_for_page": 0,
        "back_read_trail": [],
    }


async def page_analyzer_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 2: 單頁循序研讀、關鍵章節動態校準與工作記憶維護節點。"""
    idx = state["current_index"]
    page_no = idx + 1
    total = state["total_pages"]
    page_img = state["image_pages"][idx]
    pending = state.get("pending_context", "")
    anchors = state.get("toc_anchors", [])
    active_anchor = _get_active_anchor(anchors, page_no)
    active_chap = active_anchor["id"] if active_anchor else state.get("active_chapter", "")
    heading_stack = state.get("heading_stack", [])
    milestones = list(state.get("chapter_milestones", []))

    logger.info(
        "[LangGraph Page] 研讀第 %d/%d 頁  所屬章節=%s (校準區間 %d~%d 頁)  前文碎片=%s",
        page_no, total, active_chap or "(無)",
        active_anchor["start_page"] if active_anchor else 1,
        active_anchor["end_page"] if active_anchor else total,
        bool(pending),
    )
    t0 = time.perf_counter()

    prompt_parts = [
        f"這是問卷的第 {page_no}/{total} 頁影像。",
    ]
    if active_anchor:
        prompt_parts.append(
            "<TOC_Navigation_Context>\n"
            f"當前物理頁碼：第 {page_no} 頁\n"
            f"當前工作章節：{active_anchor['id']} - {active_anchor.get('title', '')}（該章節真實物理起始頁：第 {active_anchor['start_page']} 頁）\n"
            f"所有章節物理錨點地圖：{json.dumps(anchors, ensure_ascii=False)}\n"
            "【提示 1】若本頁頂部/中段出現了新章節大標題，請填寫 detected_new_chapter 進行動態校準！\n"
            "【提示 2】若本頁題目丟失父級定義，可指定 back_read_pages 回讀真實起始頁！\n"
            "</TOC_Navigation_Context>"
        )
    if heading_stack:
        prompt_parts.append(f"【當前活躍標題棧】：{' -> '.join(heading_stack)}")
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
        cleaned = _clean_json(raw_output)
        res = json.loads(cleaned)
        if not isinstance(res, dict):
            res = {}
    except Exception as e:
        logger.warning("[LangGraph Page警告] 第 %d 頁解析失敗：%s", page_no, e)
        res = {}

    extracted = _validate_extracted_questions(res.get("extracted_questions", []))
    new_pending = str(res.get("pending_context", "")).strip()
    detected_chap = res.get("detected_new_chapter")
    curr_heading = str(res.get("current_heading", "")).strip()
    request_back_read = bool(res.get("request_back_read", False))
    raw_back_pages = res.get("back_read_pages", [])

    # 1. 執行關鍵頁面章節動態校準
    calibrated_anchors = anchors
    if detected_chap and isinstance(detected_chap, dict) and detected_chap.get("id"):
        calibrated_anchors, is_calibrated, m_msg = _calibrate_toc_anchors(
            anchors=anchors,
            current_page=page_no,
            detected_chapter=detected_chap,
            total_pages=total,
        )
        if is_calibrated and m_msg:
            milestones.append(m_msg)
            # 重置 active_chapter 與標題棧
            active_chap = str(detected_chap["id"]).strip()
            heading_stack = [active_chap]

    new_active_chap = str(res.get("active_chapter", active_chap)).strip() or active_chap

    # 2. 規範化回讀頁碼列表（過濾合法頁碼，排除當前頁與超出範圍頁）
    back_read_pages: list[int] = []
    if isinstance(raw_back_pages, list):
        for p in raw_back_pages:
            try:
                p_int = int(p)
                if 1 <= p_int < page_no:
                    back_read_pages.append(p_int)
            except (ValueError, TypeError):
                continue
    # 若 LLM 請求回讀但未指定頁碼，預設為前一頁
    if request_back_read and not back_read_pages and page_no > 1:
        back_read_pages = [page_no - 1]

    # 3. 更新標題棧
    new_stack = list(heading_stack)
    if new_active_chap and (not new_stack or new_stack[0] != new_active_chap):
        new_stack = [new_active_chap]
    if curr_heading and curr_heading not in new_stack:
        new_stack.append(curr_heading)

    logger.info(
        "[LangGraph Page完成] 第 %d/%d 頁  萃取題目=%d 筆  新碎片=%s  請求回讀=%s (目標頁=%s)  耗時=%.1f ms",
        page_no, total, len(extracted), bool(new_pending), request_back_read, back_read_pages, (time.perf_counter() - t0) * 1000,
    )

    history = dict(state.get("page_history", {}))
    history[page_no] = PageAnalysisResult(
        page_number=page_no,
        detected_new_chapter=detected_chap if isinstance(detected_chap, dict) else None,
        extracted_questions=extracted,
        pending_context=new_pending,
        active_chapter=new_active_chap,
        current_heading=curr_heading,
        request_back_read=request_back_read,
        back_read_pages=back_read_pages,
    )

    return {
        "toc_anchors": calibrated_anchors,
        "pending_context": new_pending,
        "active_chapter": new_active_chap,
        "heading_stack": new_stack,
        "page_history": history,
        "chapter_milestones": milestones,
        "requested_back_read_pages": back_read_pages,
        "is_back_reading": request_back_read and bool(back_read_pages),
        "loop_count": state.get("loop_count", 0) + 1,
    }


async def back_read_inspector_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 4: 支援任意指定多頁的記憶回讀審查節點（對照校準後的真實物理起始頁）。"""
    idx = state["current_index"]
    page_no = idx + 1
    total = state["total_pages"]
    target_pages = state.get("requested_back_read_pages", [])
    if not target_pages:
        target_pages = [max(1, page_no - 1)]

    # 限制最多打包 3 張關聯頁（排序去重）
    target_pages = sorted(list(set(target_pages)))[-2:]
    trail_msg = f"Page {page_no} -> Back-read to real calibrated pages {target_pages}"
    logger.info("[LangGraph Back-Read] 啟動多頁跨度記憶回讀：調取真實物理頁面 %s 與當前第 %d 頁影像進行對照...", target_pages, page_no)
    t0 = time.perf_counter()

    images_to_send: list[dict[str, str]] = []
    history = state.get("page_history", {})
    history_summary = []

    for p in target_pages:
        p_idx = p - 1
        if 0 <= p_idx < len(state["image_pages"]):
            images_to_send.append(state["image_pages"][p_idx])
            p_rec = history.get(p, {})
            history_summary.append({
                "page": p,
                "chapter": p_rec.get("active_chapter", ""),
                "last_heading": p_rec.get("current_heading", ""),
                "extracted_sample": p_rec.get("extracted_questions", [])[:3],
            })

    images_to_send.append(state["image_pages"][idx])

    prompt = (
        f"這是問卷關聯真實物理頁面 {target_pages} 與當前第 {page_no} 頁的連續對照影像。\n"
        f"關聯頁面歷史快照：{json.dumps(history_summary, ensure_ascii=False)}\n"
        f"請對照前置頁的章節標題與父級編號，精確還原第 {page_no} 頁的所有正確題號與題目內容。\n"
    )

    try:
        raw_output = await chat_completion_vision(
            prompt=prompt,
            images=images_to_send,
            system_prompt=LANGGRAPH_BACK_READ_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=4096,
        )
        cleaned = _clean_json(raw_output)
        res = json.loads(cleaned)
        if not isinstance(res, dict):
            res = {}
    except Exception as e:
        logger.warning("[LangGraph Back-Read警告] 多頁回讀對照失敗：%s", e)
        res = {}

    extracted = _validate_extracted_questions(res.get("extracted_questions", []))
    new_pending = str(res.get("pending_context", "")).strip()

    hist_copy = dict(history)
    hist_copy[page_no] = PageAnalysisResult(
        page_number=page_no,
        detected_new_chapter=None,
        extracted_questions=extracted,
        pending_context=new_pending,
        active_chapter=state.get("active_chapter", ""),
        current_heading=res.get("current_heading", ""),
        request_back_read=False,
        back_read_pages=[],
    )

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
    """推進至下一頁。"""
    return {
        "current_index": state["current_index"] + 1,
        "is_back_reading": False,
        "requested_back_read_pages": [],
        "back_read_count_for_page": 0,
    }


async def final_reducer_node(state: QuestionnaireState) -> QuestionnaireState:
    """Node 5: 全局統整、跨頁去重與題號統一規範化節點。"""
    history = state.get("page_history", {})
    all_extracted: list[dict] = []
    for p in sorted(history.keys()):
        all_extracted.extend(history[p].get("extracted_questions", []))

    logger.info("[LangGraph Reducer] 匯總各頁題目，初始總筆數=%d 筆...", len(all_extracted))
    before_count = len(all_extracted)
    deduped = _canonicalize_all_question_ids(all_extracted)
    logger.info("[LangGraph Reducer] 跨頁去重與題號規範化完成：%d 筆 -> %d 筆", before_count, len(deduped))

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

    # 最終輸出再做一次題號格式規範化保障
    final_canonical = _canonicalize_all_question_ids(deduped)
    return {"final_questions": final_canonical}


# ═══════════════════════════════════════════════════════════════════════════
#  條件路由 (Conditional Routing Logic)
# ═══════════════════════════════════════════════════════════════════════════

def page_routing_decision(state: QuestionnaireState) -> str:
    """
    決定單頁研讀後的下一步：
      - 'back_read': 觸發指定頁記憶回讀
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
    """建構支援目錄動態校準與任意頁回讀的 LangGraph 狀態機。"""
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
    使用 LangGraph 擬人化循序閱讀、關鍵頁動態校準與任意頁記憶回讀狀態機解析問卷。
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
        "toc_anchors": [],
        "active_chapter": "",
        "heading_stack": [],
        "pending_context": "",
        "page_history": {},
        "chapter_milestones": [],
        "final_questions": [],
        "requested_back_read_pages": [],
        "is_back_reading": False,
        "back_read_count_for_page": 0,
        "back_read_trail": [],
        "loop_count": 0,
    }

    graph = build_questionnaire_graph()
    config = {"configurable": {"thread_id": f"breakdown_{int(time.time()*1000)}"}}

    # 3. 執行狀態機
    final_output = await graph.ainvoke(initial_state, config=config)
    results = final_output.get("final_questions", [])

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    milestones = final_output.get("chapter_milestones", [])
    if milestones:
        logger.info("[LangGraph 關鍵章節校準里程碑] %s", " | ".join(milestones))
    trail = final_output.get("back_read_trail", [])
    if trail:
        logger.info("[LangGraph 回讀翻閱軌跡] %s", " | ".join(trail))
    logger.info("[LangGraph 結束] 最終萃取題目數=%d 筆  總耗時=%.1f ms", len(results), total_ms)
    return results
