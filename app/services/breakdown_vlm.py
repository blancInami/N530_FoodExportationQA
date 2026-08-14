"""
Breakdown VLM: 視覺多模態問卷解析服務（優化版）。

核心特性：
  1. 視覺全域骨架掃描 (Pass 1: Visual Skeleton)：
     從問卷前置頁或各頁大標題建立 Outline Tree，為每頁計算 Active Ancestor Path。
  2. 帶祖先路徑 + 題目性判定之並行萃取 (Pass 2: Question-Aware Map)：
     在 Prompt 中明確定義「何為實質問卷題目」，嚴格剔除表格標題、填寫說明、法規引文等偽題目。
  3. 全域去重 + 偽題目二次審查 (Pass 3: Global Reducer with Question Validation)：
     跨頁文字無縫拼接，由 LLM 審核全局題號格式並剔除非問卷雜訊。
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.lo.file_utils import expand_to_image_pages
from app.services.llm import chat_completion_vision, chat_completion

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Prompts: 視覺大綱掃描、題目萃取與全局審查
# ═══════════════════════════════════════════════════════════════════════════

VLM_SKELETON_SYSTEM_PROMPT = """\
你是一個精準的文件大綱結構分析專家。
你的任務是根據輸入的文件頁面影像或標題資訊，梳理並建立整份問卷的「規範章節大綱樹（Document Outline Tree）」。

【執行規則】
1. 階層辨識：識別出所有主章節與主要標題（如 Part A, Chapter I, 1. General, 2.2 Standards 等）。
2. 編號標準化（Canonical ID）：為每個章節節點賦予規範編號（如 "I", "I.1", "Part_A.1"）。
3. 頁碼關聯：標註該章節首次出現的頁碼 (1-based)。

【輸出約束】
- 輸出合法的 JSON Array of Objects：
  [
    {
      "canonical_id": "章節規範編號 (如 I, II, Part_A, 1)",
      "heading_title": "章節名稱 (純英文)",
      "page_number": 1
    }
  ]
- 絕對禁止包含任何解釋文字或 Markdown code fence。\
"""

VLM_EXTRACTION_SYSTEM_PROMPT = """\
你是一個精準的視覺多模態問卷資料工程解析器（VLM Parser）。
你的任務是根據輸入的文件頁面影像，精確萃取出該頁面中的所有「問卷調查實質題目」，嚴格排除非題目雜訊。

【重要：何為問卷題目 vs 何為雜訊】
✅ 必須萃取的問卷題目（Questions / Requirements）：
  - 要求受查國主管機關回答「是否（Yes/No）」、「請說明（Please describe/explain）」、「請提供數據/法規（Please provide）」之具體問卷問題。
  - 要求填寫具體作業流程、控制措施、監測計畫之調查項目。
  - 表格中需要回答的每一個具體細項/子要求。

❌ 嚴格剔除的非題目雜訊（Non-Question Noise，絕對不要輸出）：
  - 表格的欄位標題與表頭（如 "No.", "Question", "Item", "Details", "Answer", "Signature", "Date"）。
  - 純填寫指引與背景說明（如 "General Instructions", "Note: Please use Annex 1..."）。
  - 純法規條文引用（如 "According to Article 5 of Law..."）。
  - 純章節容器大標題（章節標題請作為題號前綴，不要單獨作為題目輸出）。
  - 審核方內部備註欄（如 "For official use only", "Evaluation score"）。

【視覺版面解析與編號規則】
1. 祖先路徑繼承（極重要）：
   - 若提供了 <Active_Ancestor_Path>（如 "II.1" 或 "Part_A"），當前頁面所有子題目的 question_id 必須以此編號前綴展開（例如 "II.1.1", "II.1.1.a"），切勿丟失主章節！
2. 視覺縮排與表格關聯：
   - 依據視覺縮排深度（Visual Indentation）判定層級（如 1. -> a. -> (i) -> 1.a.i）。
   - 仔細比對表格左欄題號與右欄題目文字。
3. 語言清洗：
   - 若題目為雙語（如日文/中文與英文對照），只萃取純英文題目內容。

【輸出約束】
- 輸出合法的 JSON Array of Objects：
  [
    {
      "question_id": "完整扁平化題號 (字串，如 1.1 或 II.1.a)",
      "question_text": "純英文題目內容 (不含題號前綴)"
    }
  ]
- 若該頁面全為純說明、目錄或無任何實質問題，請回傳空陣列 []。
- 絕對禁止在 JSON 前後加上任何解釋性文字或 Markdown 標籤。\
"""

VLM_CONSISTENCY_SYSTEM_PROMPT = """\
你是資料品質檢核員。你收到的是從一份問卷多個頁面中透過視覺多模態模型萃取並匯總的題目清單。
你的任務是進行全局一致性校驗，並徹底過濾誤抓的非問卷雜訊。

【檢核規則】
1. 偽題目與雜訊過濾（核心任務）：
   - 檢查每筆條目是否真的是「問卷問題」。
   - 若條目實際上只是表格表頭（如 "Detailed Questions"）、填寫指南、簽名欄或純章節標題，請直接刪除該條目！
2. 跨頁重複合併：若有跨頁切分導致相同 question_id 的多筆條目，合併其 question_text（以空格連接）。
3. 題號層級校正：確保整體題號格式統一（點分十進位），修正可能的斷鏈。
4. 穩定性原則：保留正確題目，切勿臆造不存在的題目。

【輸出】
修正與過濾後的 JSON Array of Objects，每個包含 "question_id" 與 "question_text"。
不得加入任何解釋性文字或 Markdown 標籤。\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  輔助結構與工具
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VlmOutlineNode:
    """視覺大綱樹節點。"""
    canonical_id: str
    heading_title: str
    page_number: int = 1


def _clean_vlm_json(raw: str) -> str:
    """剝離 Markdown code fence 與首尾空白。"""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned.strip()


def _validate_vlm_extraction(data: object) -> list[dict]:
    """驗證 VLM 輸出的合法 JSON Array 格式。"""
    if not isinstance(data, list):
        raise ValueError(f"VLM 回傳非 JSON Array，實際類型：{type(data).__name__}")
    validated: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if "question_id" not in item or "question_text" not in item:
            continue
        qid = str(item["question_id"]).strip()
        qtext = str(item["question_text"]).strip()
        if qid and qtext:
            validated.append({"question_id": qid, "question_text": qtext})
    return validated


def _deduplicate_items(items: list[dict]) -> list[dict]:
    """合併跨頁或相同 question_id 的題目文本。"""
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
#  Pass 1: 視覺全域骨架掃描 (Visual Skeleton Extraction)
# ═══════════════════════════════════════════════════════════════════════════

async def _extract_vlm_document_skeleton(pages: list[dict[str, str]]) -> list[VlmOutlineNode]:
    """
    Pass 1: 從問卷前 3 頁或關鍵頁面影像中快速掃描全域章節結構。
    """
    total_pages = len(pages)
    if total_pages <= 1:
        return []

    # 選取前 3 頁（通常包含目錄與主要章節劃分）
    sample_pages = pages[:min(3, total_pages)]
    logger.info("[VLM Pass 1] 啟動視覺大綱骨架掃描（採樣前 %d 頁）...", len(sample_pages))
    t0 = time.perf_counter()

    prompt = (
        f"這是一份共 {total_pages} 頁問卷的前 {len(sample_pages)} 頁。"
        "請快速分析整份問卷的主章節架構，輸出章節大綱樹 (Outline Tree)。\n"
    )

    try:
        raw_output = await chat_completion_vision(
            prompt=prompt,
            images=sample_pages,
            system_prompt=VLM_SKELETON_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=2048,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        cleaned = _clean_vlm_json(raw_output)
        data = json.loads(cleaned)
        nodes: list[VlmOutlineNode] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "canonical_id" in item:
                    nodes.append(VlmOutlineNode(
                        canonical_id=str(item["canonical_id"]).strip(),
                        heading_title=str(item.get("heading_title", "")).strip(),
                        page_number=int(item.get("page_number", 1)),
                    ))
        logger.info("[VLM Pass 1完成] 成功建立全域大綱樹，共 %d 個章節節點  耗時=%.1f ms", len(nodes), elapsed)
        return nodes
    except Exception as e:
        logger.warning("[VLM Pass 1跳過] 視覺大綱掃描失敗，回退至局部頁面推論：%s", e)
        return []


def _bind_page_ancestor_paths(total_pages: int, skeleton_nodes: list[VlmOutlineNode]) -> list[str]:
    """
    根據全域大綱節點的起始頁碼，為每頁計算其活躍的祖先章節路徑。
    回傳 list of ancestor_path 字串（長度為 total_pages）。
    """
    page_paths = [""] * total_pages
    if not skeleton_nodes:
        return page_paths

    sorted_nodes = sorted(skeleton_nodes, key=lambda n: n.page_number)
    curr_node = sorted_nodes[0]
    node_idx = 0

    for page_i in range(total_pages):
        page_no = page_i + 1
        while node_idx + 1 < len(sorted_nodes) and sorted_nodes[node_idx + 1].page_number <= page_no:
            node_idx += 1
            curr_node = sorted_nodes[node_idx]
        page_paths[page_i] = curr_node.canonical_id

    return page_paths


# ═══════════════════════════════════════════════════════════════════════════
#  Pass 2: 帶祖先路徑與題目性判定之並行萃取
# ═══════════════════════════════════════════════════════════════════════════

async def _extract_single_page(
    page_img: dict[str, str],
    page_index: int,
    total_pages: int,
    ancestor_path: str = "",
) -> list[dict]:
    """Pass 2: 單頁視覺多模態萃取（注入祖先路徑與題目性嚴格限制）。"""
    page_no = page_index + 1
    logger.info(
        "[VLM頁面推論開始] 第 %d/%d 頁  祖先路徑=%s",
        page_no, total_pages, ancestor_path or "(無)",
    )
    t0 = time.perf_counter()

    prompt_parts = [
        f"請仔細閱讀此問卷第 {page_no}/{total_pages} 頁的版面影像，"
        "嚴格依照指示只提取出所有「實質問卷題目」，排除表頭與說明雜訊。\n"
    ]
    if ancestor_path:
        prompt_parts.append(
            "<Active_Ancestor_Path>\n"
            f"當前頁面處於主章節：{ancestor_path}\n"
            f"【重要】此頁所有子題目之 question_id 必須以 '{ancestor_path}' 前綴展開（例如 {ancestor_path}.1），切勿遺漏主章節！\n"
            "</Active_Ancestor_Path>\n"
        )
    prompt = "\n".join(prompt_parts)

    raw_output = await chat_completion_vision(
        prompt=prompt,
        images=[page_img],
        system_prompt=VLM_EXTRACTION_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=4096,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    cleaned = _clean_vlm_json(raw_output)
    try:
        parsed = json.loads(cleaned)
        items = _validate_vlm_extraction(parsed)
        logger.info(
            "[VLM頁面推論完成] 第 %d/%d 頁  萃取題目=%d 筆  耗時=%.1f ms",
            page_no, total_pages, len(items), elapsed,
        )
        return items
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "[VLM頁面推論警告] 第 %d/%d 頁 JSON 解析失敗：%s  耗時=%.1f ms",
            page_no, total_pages, e, elapsed,
        )
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  主公開介面 (extract_questions_from_file_vlm)
# ═══════════════════════════════════════════════════════════════════════════

async def extract_questions_from_file_vlm(file_bytes: bytes, filename: str) -> list[dict]:
    """
    使用視覺多模態 (VLM) 深度優化引擎從文件位元組中萃取問卷題目。
    """
    settings = get_settings()
    pipeline_start = time.perf_counter()
    mime = "application/pdf" if filename.lower().endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    logger.info("[VLM管線啟動] 檔名=%s  大小=%.2f KB  併發上限=%d", filename, len(file_bytes) / 1024, settings.vlm_max_concurrency)
    
    # ── 1. 展開為頁面影像 ────────────────────────────────────────────────
    logger.info("[VLM管線步驟1/4] 開始調用 LibreOffice/Poppler 進行頁面展開...")
    t_conv_start = time.perf_counter()
    pages = await expand_to_image_pages(
        raw=file_bytes,
        mime_type=mime,
        filename=filename,
    )
    if not pages:
        raise ValueError(f"無法將檔案 {filename} 轉換為有效頁面影像")

    total_pages = len(pages)
    t_conv_elapsed = (time.perf_counter() - t_conv_start) * 1000
    logger.info("[VLM管線步驟1/4完成] 頁面轉換成功，共 %d 頁  耗時=%.1f ms", total_pages, t_conv_elapsed)

    # ── 2. Pass 1 視覺全域骨架掃描與祖先路徑計算 ─────────────────────────
    skeleton_nodes: list[VlmOutlineNode] = []
    if total_pages > 1:
        skeleton_nodes = await _extract_vlm_document_skeleton(pages)
    page_ancestor_paths = _bind_page_ancestor_paths(total_pages, skeleton_nodes)

    # ── 3. Pass 2 多頁並行 VLM 推論 ──────────────────────────────────────
    logger.info("[VLM管線步驟2/4] 開始進行多頁並行 VLM 推論（共 %d 頁）...", total_pages)
    sem = asyncio.Semaphore(settings.vlm_max_concurrency)

    async def _guarded_page(idx: int) -> list[dict]:
        async with sem:
            return await _extract_single_page(pages[idx], idx, total_pages, page_ancestor_paths[idx])

    tasks = [_guarded_page(i) for i in range(total_pages)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── 4. 合併各頁結果與去重 ────────────────────────────────────────────
    logger.info("[VLM管線步驟3/4] 匯總各頁題目並進行跨頁去重與文字合併...")
    merged: list[dict] = []
    success_pages = 0
    fail_pages = 0
    for i, r in enumerate(results):
        if isinstance(r, list):
            success_pages += 1
            merged.extend(r)
        elif isinstance(r, BaseException):
            fail_pages += 1
            logger.warning("[VLM頁面異常] 第 %d 頁處理失敗：%s", i + 1, r)

    logger.info("[VLM頁面推論統計] 成功頁數=%d  失敗頁數=%d  初步匯總題目=%d 筆", success_pages, fail_pages, len(merged))

    if not merged:
        raise ValueError(f"VLM 解析 {filename} 失敗：所有頁面皆未產出有效題目")

    before_dedup_count = len(merged)
    merged = _deduplicate_items(merged)
    logger.info("[VLM管線步驟3/4完成] 去重合併完成：原始=%d 筆 -> 去重後=%d 筆", before_dedup_count, len(merged))

    # ── 5. Pass 3 全局一致性審查與偽題目二次過濾 ─────────────────────────
    if len(merged) <= 150:
        logger.info("[VLM管線步驟4/4] 啟動全局一致性校驗與偽題目過濾 (題目數=%d 筆)...", len(merged))
        t_verify_start = time.perf_counter()
        try:
            prompt = json.dumps(merged, ensure_ascii=False, indent=2)
            raw_v = await chat_completion(
                prompt=prompt,
                system_prompt=VLM_CONSISTENCY_SYSTEM_PROMPT,
                temperature=0,
                max_tokens=8192,
            )
            verified = json.loads(_clean_vlm_json(raw_v))
            verified_items = _validate_vlm_extraction(verified)
            if verified_items:
                logger.info("[VLM管線步驟4/4完成] 全局校驗成功：校驗前=%d 筆 -> 校驗後=%d 筆", len(merged), len(verified_items))
                merged = verified_items
        except Exception as e:
            logger.warning("[VLM管線步驟4/4跳過] 全局一致性審查未套用：%s", e)
        t_verify_elapsed = (time.perf_counter() - t_verify_start) * 1000
        logger.info("[VLM全局校驗耗時] %.1f ms", t_verify_elapsed)
    else:
        logger.info("[VLM管線步驟4/4跳過] 題目數超過 150 筆，略過全局校驗")

    total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info("[VLM全管線結束] 檔名=%s  最終輸出題目數=%d 筆  總耗時=%.1f ms", filename, len(merged), total_pipeline_ms)
    return merged
