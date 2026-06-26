"""
QA Router: /api/v1/qa/ask, /api/v1/qa/ingest, /api/v1/qa/ingest-legal, /api/v1/qa/breakdown
"""
import asyncio
import logging
import os
import re
import tempfile
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db_session
from app.schemas.breakdown import BreakdownItem
from app.schemas.qa import (
    AskRequest, AskResponse, ReferenceSource, KnowledgeReferenceSource,
    IngestRequest, IngestResponse, IngestItemResult,
)
from app.services.breakdown import convert_file_to_markdown, extract_questions
from app.services.intent import get_intent_classifier
from app.services.retrieval import hybrid_retrieve
from app.services.llm import chat_completion
from app.services.ingest import ingest_questionnaire
from app.services.query_log import save_query_log
from app.utils.dictionary import DictionaryMatcher, build_dictionary_xml, get_dictionary_matcher
from app.utils.opencc_converter import s2t
from app.utils.connection_manager import monitor_disconnect, run_interruptible

router = APIRouter(prefix="/api/v1/qa", tags=["QA"])
logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "xlsx", "xls"}
_ENRICHMENT_TERM_LIMIT = 5


def _build_qa_prompt(query: str, context: str, dictionary_xml: str, knowledge_context: str = "") -> str:
    """
    Build the structured XML prompt for the LLM.
    Injects questionnaire QA context, optional knowledge document references
    (REGULATION / GUIDELINE / QA), and the filtered mandatory dictionary.
    """
    knowledge_block = ""
    if knowledge_context:
        knowledge_block = f"""\n<Knowledge_References>
{knowledge_context}
</Knowledge_References>
"""
    return f"""<Historical_QA>
{context}
</Historical_QA>
{knowledge_block}
{dictionary_xml}

<User_Query>
{query}
</User_Query>

<Output_Instructions>
Based on the historical QA context above, answer the user's query.
You MUST use the exact English terms from <Mandatory_Dictionary> for any matching concepts.
If <Knowledge_References> is provided, cite the relevant knowledge content accurately — use official text verbatim where available.

Output format (strictly follow):
1. First, produce a COMPLETE English answer.
2. Then output a separator line: ---
3. Then produce an ACCURATE Traditional Chinese (正體中文) translation of the English answer.

Do NOT add any other sections or formatting beyond this structure.
</Output_Instructions>"""


def _parse_llm_output(output: str) -> tuple[str, str]:
    """Parse LLM output into (english_reply, chinese_reply)."""
    parts = re.split(r"\n-{3,}\n", output, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    parts = output.split("---", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return output.strip(), output.strip()


def _is_chinese(text: str) -> bool:
    """Heuristic: check if text contains significant Chinese characters."""
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return chinese_chars > len(text) * 0.3


async def _translate_to_chinese(
    text: str,
    matcher: DictionaryMatcher | None = None,
    interrupt_event: asyncio.Event | None = None,
) -> str:
    """Translate a non-Chinese text into Traditional Chinese using the LLM."""
    logger.info("將輸入翻譯為繁體中文：輸入長度=%d", len(text))

    # 從官方正規詞彙比對英文術語，約束翻譯詞彙一致性
    terminology_block = ""
    if matcher is not None:
        term_hits = matcher.filter_by_english_text(text)
        if term_hits:
            entries = "\n".join(f"  - {en} → {zh}" for zh, en in term_hits.items())
            terminology_block = (
                f"\n\nOfficial terminology mapping "
                f"(MUST use these exact Traditional Chinese terms):\n{entries}\n"
            )
            logger.info("翻譯術語約束：命中 %d 筆官方術語", len(term_hits))

    prompt = (
        "You are a professional translator. Translate the following text into Traditional Chinese (正體中文). "
        "Output ONLY the Chinese translation, nothing else."
        f"{terminology_block}\n\n"
        f"{text}"
    )
    result = await chat_completion(prompt, temperature=0, max_tokens=1024, interrupt_event=interrupt_event)
    translated = s2t(result.strip())
    logger.info("翻譯為繁體中文完成：輸出長度=%d", len(translated))
    return translated


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="提交問題（POST）",
    description=(
        "以 JSON body 提交食品輸銷相關問題，執行完整 RAG 問答管線，回傳雙語解答與參考來源。\n\n"
        "**管線流程：**\n\n"
        "1. **Phase 1a — 語言偵測與翻譯**：若輸入非中文，先以 LLM 翻譯為繁體中文。\n"
        "2. **Phase 1b — 意圖分類（雙層）**：\n"
        "   - Layer 1：Aho-Corasick 比對「單位對照表」分工關鍵字，快速判定負責機關/單位。\n"
        "   - Layer 2：Layer 1 無命中時，以 LLM 兜底分類。\n"
        "3. **Phase 1c — 術語特徵擴增**：比對官方正規詞彙（6000+ 筆），命中前 5 筆術語注入查詢以提升向量搜尋精度。\n"
        "4. **Phase 2 — 雙軌混合檢索**：\n"
        "   - **Track A（問卷庫）**：對「問卷題目切塊」執行 cosine distance 向量搜尋，展開相鄰切塊後彙整參考來源。\n"
        "   - **Track B（知識文獻庫）**：對「文獻切塊檔」執行向量搜尋，命中結果依文獻類型（REGULATION／GUIDELINE／QA）格式化。\n"
        "   - 兩軌合併策略由環境變數 `RETRIEVAL_MERGE_MODE` 控制（`independent`：各自保留 top_n；`compete`：混排共用名額）。\n"
        "5. **Phase 3 — LLM 雙語生成**：以歷史問答 context、知識文獻 context 及術語字典組裝 prompt，呼叫 LLM 生成英文與正體中文回覆。\n"
        "6. **Phase 4 — 回應組裝**：扁平化回傳負責單位、雙語回覆、附件超連結及所有參考來源。\n\n"
        "**注意事項：**\n"
        "- `similarity_threshold`：向量相似度閾值（距離，越小越相似），預設使用系統環境變數 `SIMILARITY_THRESHOLD`（0.75）。\n"
        "- `top_n`：最終聚合取回筆數，預設使用系統環境變數 `TOP_N`（5）。"
    ),
    response_description="雙語問答結果，包含負責單位、中英回覆、附件超連結與雙軌參考來源。",
    responses={
        200: {"description": "問答成功，回傳 AskResponse"},
        422: {"description": "請求格式驗證失敗（如 question 超長或參數超出範圍）"},
        500: {"description": "系統內部錯誤（LLM 或向量搜尋異常）"},
    },
)
async def ask_question_post(
    body: AskRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> AskResponse:
    return await _process_ask(
        body.question,
        request,
        session,
        similarity_threshold=body.similarity_threshold,
        top_n=body.top_n,
        intent_top_n=body.intent_top_n,
    )


@router.get(
    "/ask",
    response_model=AskResponse,
    summary="提交問題（GET）",
    description=(
        "以 Query String 提交食品輸銷相關問題，執行與 `POST /ask` 相同的 RAG 問答管線，"
        "適合瀏覽器直接測試或輕量整合場景。\n\n"
        "**管線流程同 `POST /ask`**，請參閱 POST 端點說明。\n\n"
        "**參數說明：**\n"
        "- `question`（必填）：問題文字，長度 1～2000 字元，支援中英文。\n"
        "- `similarity_threshold`（選填）：向量搜尋距離閾值（0.0～1.0），未帶入則使用系統預設值。\n"
        "- `top_n`（選填）：最終聚合取回筆數（1～100），未帶入則使用系統預設值。\n"
        "- `intent_top_n`（選填）：意圖分類回傳的負責單位最高數量（1～10），未帶入則使用系統預設值。"
    ),
    response_description="雙語問答結果，包含負責單位、中英回覆、附件超連結與雙軌參考來源。",
    responses={
        200: {"description": "問答成功，回傳 AskResponse"},
        422: {"description": "請求格式驗證失敗（如 question 超長或參數超出範圍）"},
        500: {"description": "系統內部錯誤（LLM 或向量搜尋異常）"},
    },
)
async def ask_question_get(
    question: str = Query(..., min_length=1, max_length=2000, description="使用者提問，支援中英文，長度 1～2000 字元"),
    similarity_threshold: float | None = Query(default=None, ge=0.0, le=1.0, description="向量搜尋距離閾值（0.0～1.0），未帶入則使用系統預設值 SIMILARITY_THRESHOLD"),
    top_n: int | None = Query(default=None, ge=1, le=100, description="最終聚合取回筆數（1～100），未帶入則使用系統預設值 TOP_N"),
    intent_top_n: int | None = Query(default=None, ge=1, le=10, description="意圖分類回傳的負責單位最高數量（1～10），未帶入則使用系統預設值 INTENT_TOP_N"),
    request: Request = ...,
    session: AsyncSession = Depends(get_db_session),
) -> AskResponse:
    return await _process_ask(
        question,
        request,
        session,
        similarity_threshold=similarity_threshold,
        top_n=top_n,
        intent_top_n=intent_top_n,
    )


async def _process_ask(
    question: str,
    request: Request,
    session: AsyncSession,
    similarity_threshold: float | None = None,
    top_n: int | None = None,
    intent_top_n: int | None = None,
) -> AskResponse:
    """
    Core QA pipeline:
    Phase 1: Translate to Chinese if needed, then intent classification
    Phase 2: Dual-path hybrid retrieval with context expansion
    Logging: fire-and-forget DB write to 查詢紀錄 after Phase 4
    Phase 3: LLM bilingual generation
    Phase 4: Structured payload response
    """
    pipeline_start = time.perf_counter()
    log_key = uuid.uuid4()
    is_chinese = _is_chinese(question)
    intent_method: str = "none"
    from app.config import get_settings
    settings = get_settings()
    actual_intent_top_n = intent_top_n or settings.intent_top_n

    logger.info(
        "QA 管線啟動：問題長度=%d  是否為中文=%s  閾値=%s  top_n=%s  intent_top_n=%d",
        len(question), is_chinese, similarity_threshold, top_n, actual_intent_top_n,
    )

    async with monitor_disconnect(request) as interrupt_signal:
        # 提前載入 DictionaryMatcher（懶載入單例，Phase 1a 翻譯與 Phase 1c/3 共用同一實例）
        matcher = await get_dictionary_matcher(session)

        # === Phase 1a: Translate non-Chinese input to Chinese ===
        if is_chinese:
            chinese_query = question
            chinese_translation = question
            logger.debug("輸入為中文，略過翻譯")
        else:
            chinese_query = await _translate_to_chinese(question, matcher, interrupt_event=interrupt_signal)
            chinese_translation = chinese_query

        # === Phase 1b: Intent Classification ===
        classifier = await get_intent_classifier(session)
        responsible_units = []
        methods = []

        # Layer 1 — Aho-Corasick (快速路徑)
        l1_intents = classifier.classify(chinese_query, top_n=actual_intent_top_n)
        if l1_intents:
            for i in l1_intents:
                responsible_units.append({"機關": i.機關 or "", "單位": i.單位 or "", "理由": i.理由})
            methods.append("aho-corasick")

        # Layer 2 — LLM Fallback (補充不足名額)
        if len(responsible_units) < actual_intent_top_n:
            remaining = actual_intent_top_n - len(responsible_units)
            l2_intents = await classifier.llm_fallback_classify(
                chinese_query, top_n=remaining, interrupt_event=interrupt_signal
            )
            if l2_intents:
                # 避免重複 (雖然 Layer 1 沒匹配到通常 Layer 2 也不會重，但做個防護)
                seen = set((u["機關"], u["單位"]) for u in responsible_units)
                added = 0
                for i in l2_intents:
                    pair = (i.機關 or "", i.單位 or "")
                    if pair not in seen:
                        responsible_units.append({"機關": i.機關 or "", "單位": i.單位 or "", "理由": i.理由})
                        seen.add(pair)
                        added += 1
                if added > 0:
                    methods.append("llm_fallback")

        # 組合分類方式標籤 (資料庫欄位長度限制為 20)
        if not methods:
            intent_method = "none"
        elif len(methods) == 1:
            intent_method = methods[0]
        else:
            intent_method = "hybrid"

        logger.info("意圖分類結果 (%s)：負責單位=%r", intent_method, responsible_units)

        # === Phase 1c: Feature Enrichment ===
        # matcher 已於 Phase 1a 前載入（懶載入單例，此處零成本複用）
        term_hits = matcher.filter_by_text(chinese_query)
        if term_hits:
            top_terms = list(term_hits.keys())[:_ENRICHMENT_TERM_LIMIT]
            top_term_str = "\u3001".join(top_terms)
            enriched_query = f"\u63d0\u554f\uff1a{chinese_query} \u76f8\u95dc\u8853\u8a9e\uff1a{top_term_str}"
            logger.info("特徵擴増：命中 %d 筆術語，注入 %d 筆", len(term_hits), len(top_terms))
        else:
            enriched_query = chinese_query
            logger.debug("特徵擴増：無命中術語，使用原始查詢")

        # === Phase 2: Hybrid Retrieval ===
        retrieval_result = await run_interruptible(
            hybrid_retrieve(
                enriched_query,
                session,
                similarity_threshold=similarity_threshold,
                top_n=top_n,
            ),
            interrupt_signal,
            label="Phase 2 向量檢索",
        )

        if not retrieval_result.chunks and not retrieval_result.knowledge_references:
            logger.warning("QA 管線：檢索無結果，回傳空白回應")
            elapsed_ms_empty = (time.perf_counter() - pipeline_start) * 1000
            asyncio.create_task(save_query_log({
                "主鍵": log_key,
                "原始問題": question,
                "是否中文": is_chinese,
                "中文譯文": chinese_translation if not is_chinese else None,
                "意圖分類方式": intent_method,
                "負責機關": ", ".join(set(u["機關"] for u in responsible_units if u.get("機關"))),
                "負責單位": ", ".join(set(u["單位"] for u in responsible_units if u.get("單位"))),
                "命中術語": term_hits or {},
                "enriched_query": enriched_query,
                "similarity_threshold": similarity_threshold,
                "top_n": top_n,
                "問卷命中數": 0,
                "知識命中數": 0,
                "參考來源": [],
                "知識參考來源": [],
                "raw_context": None,
                "knowledge_context": None,
                "dictionary_xml": None,
                "llm_prompt": None,
                "llm_output": None,
                "英文回覆": "Sorry, there is currently no relevant data in the database to answer this question.",
                "中文回覆": "抱歉，目前資料庫中尚無相關資料可供回答此問題。",
                "耗時毫秒": elapsed_ms_empty,
            }))
            return AskResponse(
                負責單位=responsible_units,
                原始題目=question,
                中文譯文=chinese_translation if not is_chinese else "",
                中文回覆="抱歉，目前資料庫中尚無相關資料可供回答此問題。",
                英文回覆="Sorry, there is currently no relevant data in the database to answer this question.",
                附件超連結=[],
                參考來源=[],
                知識參考來源=[],
            )

        logger.info(
            "檢索結果：%d 個問卷切塊  %d 個知識切塊  %d 筆參考來源  %d 筆附件",
            len(retrieval_result.chunks),
            len(retrieval_result.knowledge_references),
            len(retrieval_result.references),
            len(retrieval_result.attachment_urls),
        )

        # === Phase 3: LLM Bilingual Generation ===
        # matcher already loaded in Phase 1c (singleton — no DB round-trip)
        # 使用 filter_by_text_grounded：剔除僅由短詞觸發且 context 無佐證的術語，防止幻覺
        filtered_dict = matcher.filter_by_text_grounded(chinese_query, retrieval_result.raw_context)
        dictionary_xml = build_dictionary_xml(filtered_dict)
        logger.info("字典篩選完成：命中 %d 筆術語（含 grounding 防護）", len(filtered_dict))
        prompt = _build_qa_prompt(
            question,
            retrieval_result.raw_context,
            dictionary_xml,
            knowledge_context=retrieval_result.knowledge_context,
        )
        logger.info("LLM 生成開始：上下文長度=%d", len(retrieval_result.raw_context))
        llm_output = await chat_completion(prompt, interrupt_event=interrupt_signal)
        english_reply, chinese_reply = _parse_llm_output(llm_output)

        chinese_reply = s2t(chinese_reply)
        if not is_chinese:
            chinese_translation = s2t(chinese_translation)

        # === Phase 4: Build Response ===
        ref_sources = [
            ReferenceSource(
                問卷主檔主鍵=r["問卷主檔主鍵"],
                問卷名稱=r["問卷名稱"],
                輸出國家=r["輸出國家"],
                輸出品項=r["輸出品項"],
                題目序號=r["題目序號"],
            )
            for r in retrieval_result.references
        ]

        knowledge_ref_sources = [
            KnowledgeReferenceSource(
                文獻主檔主鍵=r["文獻主檔主鍵"],
                文獻名稱=r["文獻名稱"],
                文獻類型=r["文獻類型"],
                節點標題路徑=r["節點標題路徑"],
            )
            for r in retrieval_result.knowledge_references
        ]

        elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            "QA 管線完成：英文回覆長度=%d  中文回覆長度=%d  來源筆數=%d  耗時=%.1f ms",
            len(english_reply), len(chinese_reply), len(ref_sources), elapsed_ms,
        )

        asyncio.create_task(save_query_log({
            "主鍵": log_key,
            "原始問題": question,
            "是否中文": is_chinese,
            "中文譯文": chinese_translation if not is_chinese else None,
            "意圖分類方式": intent_method,
            "負責機關": ", ".join(set(u["機關"] for u in responsible_units if u.get("機關"))),
            "負責單位": ", ".join(set(u["單位"] for u in responsible_units if u.get("單位"))),
            "命中術語": term_hits,
            "enriched_query": enriched_query,
            "similarity_threshold": similarity_threshold,
            "top_n": top_n,
            "問卷命中數": len(retrieval_result.chunks),
            "知識命中數": len(retrieval_result.knowledge_references),
            "參考來源": [r.model_dump() for r in ref_sources],
            "知識參考來源": [r.model_dump() for r in knowledge_ref_sources],
            "raw_context": retrieval_result.raw_context,
            "knowledge_context": retrieval_result.knowledge_context,
            "dictionary_xml": dictionary_xml,
            "llm_prompt": prompt,
            "llm_output": llm_output,
            "英文回覆": english_reply,
            "中文回覆": chinese_reply,
            "耗時毫秒": elapsed_ms,
        }))

        return AskResponse(
            負責單位=responsible_units,
            原始題目=question,
            中文譯文=chinese_translation if not is_chinese else "",
            中文回覆=chinese_reply,
            英文回覆=english_reply,
            附件超連結=retrieval_result.attachment_urls,
            參考來源=ref_sources,
            知識參考來源=knowledge_ref_sources,
        )



@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="問卷向量化匯入",
    description=(
        "指定一或多筆「問卷主檔主鍵」，觸發問卷題目的切塊向量化流程，並將結果寫入「問卷題目切塊」資料表，"
        "供後續 `/ask` 端點的向量相似度搜尋使用。\n\n"
        "**處理流程：**\n\n"
        "1. 依主鍵從「問卷題目檔」取出題目與回覆原文。\n"
        "2. 對每筆題目執行滑動視窗切塊（`CHUNK_SIZE`／`CHUNK_OVERLAP` 由環境變數控制）。\n"
        "3. 批次呼叫 Embedding Server，取得每個切塊的 1024 維向量。\n"
        "4. 使用 `sanitize_text_for_db()` 清除控制字元後，批次寫入資料庫。\n\n"
        "**注意事項：**\n"
        "- 若問卷主鍵不存在，該筆回傳 `status: not_found`，不影響其他筆處理。\n"
        "- 舊有切塊會先被刪除後重新寫入（重新匯入等冪）。\n"
        "- 回傳統計包含成功處理的題目數與切塊總數。"
    ),
    response_description="批次匯入統計摘要，包含問卷數、題目數與切塊寫入總數。",
    responses={
        200: {"description": "匯入流程完成（部分失敗時各筆 status 為 error）"},
        422: {"description": "請求格式驗證失敗（如主鍵清單為空）"},
        500: {"description": "系統內部錯誤（Embedding Server 或 DB 異常）"},
    },
)
async def ingest_questionnaire_endpoint(
    body: IngestRequest,
    session: AsyncSession = Depends(get_db_session),
) -> IngestResponse:
    pks = body.問卷主檔主鍵清單
    logger.info("收到匯入請求：共 %d 份問卷", len(pks))

    results: list[IngestItemResult] = []
    total_items = 0
    total_chunks = 0

    for pk in pks:
        try:
            raw = await ingest_questionnaire(pk, session)
            item = IngestItemResult(
                問卷主檔主鍵=pk,
                status=raw.get("status", "success"),
                items_processed=raw.get("items_processed", 0),
                total_chunks_written=raw.get("total_chunks_written", 0),
                message=raw.get("message", ""),
            )
            logger.info(
                "問卷匯入結果：pk=%s  狀態=%s  題目數=%d  切塊數=%d",
                pk, item.status, item.items_processed, item.total_chunks_written,
            )
        except Exception as e:
            logger.error("問卷匯入失敗：pk=%s  錯誤=%s", pk, e, exc_info=True)
            item = IngestItemResult(
                問卷主檔主鍵=pk,
                status="error",
                message=str(e),
            )
        results.append(item)
        total_items += item.items_processed
        total_chunks += item.total_chunks_written

    logger.info(
        "匯入端點完成：%d 份問卷  %d 筆題目  %d 個切塊",
        len(results), total_items, total_chunks,
    )
    return IngestResponse(
        total_questionnaires=len(results),
        total_items_processed=total_items,
        total_chunks_written=total_chunks,
        results=results,
    )


@router.post(
    "/breakdown",
    response_model=list[BreakdownItem],
    summary="問卷檔案題目萃取",
    description=(
        "上傳問卷原始檔案，透過 MarkItDown 轉換與 LLM 語義萃取，自動識別並輸出結構化題號與英文題目清單。\n\n"
        "**支援格式：** `pdf`、`doc`、`docx`、`xlsx`、`xls`\n\n"
        "**處理管線：**\n\n"
        "1. **副檔名驗證**：不支援的格式回傳 HTTP 415。\n"
        "2. **寫入暫存檔**：上傳內容寫入系統暫存路徑。\n"
        "3. **檔案轉換**：以 MarkItDown 將文件轉換為 Markdown 文字（非同步執行，不阻塞事件循環）。\n"
        "4. **LLM 語義萃取**：滑動視窗切塊後，使用 LLM 識別「點分十進位題號」與「純英文題目文字」。\n"
        "5. **清理暫存檔**：無論成功或失敗均自動刪除暫存檔。\n\n"
        "**題號格式範例：** `1`、`1.1`、`2.2.1`、`1.2.4.a`\n\n"
        "**注意事項：**\n"
        "- 若 LLM 解析失敗，回傳 HTTP 502。\n"
        "- 上傳檔案大小建議不超過 50 MB，否則處理時間可能較長。"
    ),
    response_description="萃取結果清單，每筆包含點分十進位題號（`question_id`）與純英文題目內容（`question_text`）。",
    responses={
        200: {"description": "萃取成功，回傳 BreakdownItem 陣列"},
        415: {"description": "不支援的檔案格式"},
        422: {"description": "Markdown 轉換失敗或格式無效"},
        502: {"description": "LLM 解析題目失敗"},
    },
)
async def breakdown_questionnaire(
    file: UploadFile,
    request: Request,
) -> list[BreakdownItem]:
    # ── 1. 驗證副檔名 ─────────────────────────────────────────────────────
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        logger.warning("Breakdown：不支援的檔案格式 — 檔名=%s  副檔名=%s", filename, ext)
        raise HTTPException(
            status_code=415,
            detail=f"不支援的檔案格式 .{ext}。允許格式：{', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    logger.info("Breakdown 端點啟動：檔名=%s  格式=%s", filename, ext)
    start = time.perf_counter()

    tmp_path: str | None = None
    try:
        # ── 2. 寫入暫存檔 ──────────────────────────────────────────────────
        content = await file.read()
        with tempfile.NamedTemporaryFile(
            suffix=f".{ext}", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        logger.debug("Breakdown：暫存檔建立 — 路徑=%s  大小=%d bytes", tmp_path, len(content))

        # ── 3. 檔案轉換 Markdown（CPU-bound → asyncio.to_thread）──────────
        async with monitor_disconnect(request) as interrupt_signal:
            # ── 3. 檔案轉換 Markdown（CPU-bound → asyncio.to_thread）
            try:
                markdown = await run_interruptible(
                    asyncio.to_thread(convert_file_to_markdown, tmp_path, ext),
                    interrupt_signal,
                    label="Breakdown Markdown 轉換",
                )
            except asyncio.CancelledError:
                raise
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

            # ── 4. LLM 萍取
            try:
                items = await run_interruptible(
                    extract_questions(markdown),
                    interrupt_signal,
                    label="Breakdown LLM 題目萍取",
                )
            except asyncio.CancelledError:
                raise
            except ValueError as e:
                raise HTTPException(status_code=502, detail=f"LLM 解析失敗：{e}")


    finally:
        # ── 5. 清理暫存檔 ──────────────────────────────────────────────────
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug("Breakdown：暫存檔已清除 — 路徑=%s", tmp_path)
            except PermissionError as e:
                logger.warning(
                    "Breakdown：暫存檔刪除失敗（可能仍在背景執行緒中被使用，稍後由系統回收）— 路徑=%s, 錯誤=%s",
                    tmp_path, e
                )
            except Exception as e:
                logger.warning("Breakdown：暫存檔刪除發生未預期錯誤 — 路徑=%s, 錯誤=%s", tmp_path, e)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Breakdown 端點完成：檔名=%s  萃取題目=%d 筆  耗時=%.1f ms",
        filename, len(items), elapsed_ms,
    )
    return [BreakdownItem(**item) for item in items]
