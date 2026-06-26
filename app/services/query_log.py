"""
Query log service: persists each /qa/ask pipeline execution to the 查詢紀錄 table.

Uses a dedicated session (not the request-scoped Depends session) to ensure the
INSERT succeeds even after the HTTP response has been returned.
Call save_query_log() as a fire-and-forget asyncio.create_task() from the router.
"""
import logging
import uuid

from sqlalchemy import insert

from app.database import get_session_factory
from app.models import 查詢紀錄
from app.utils import sanitize_text_for_db

logger = logging.getLogger(__name__)


async def save_query_log(log_data: dict) -> None:
    """
    Insert one row into 查詢紀錄.

    All exceptions are caught and logged — this function must never raise,
    as it is called fire-and-forget after the response has been sent.
    """
    factory = get_session_factory()
    if factory is None:
        logger.warning("查詢紀錄寫入略過：session factory 尚未初始化")
        return

    try:
        async with factory() as session:
            async with session.begin():
                row = _build_row(log_data)
                await session.execute(insert(查詢紀錄).values(**row))
        logger.debug("查詢紀錄寫入完成：主鍵=%s", log_data.get("主鍵"))
    except Exception:
        logger.exception("查詢紀錄寫入失敗（不影響主流程）")


def _build_row(d: dict) -> dict:
    """Sanitize text fields and build the INSERT value dict."""
    def _s(v: str | None) -> str | None:
        return sanitize_text_for_db(v) if isinstance(v, str) else v

    return {
        "主鍵": d.get("主鍵") or uuid.uuid4(),
        "原始問題": _s(d.get("原始問題", "")),
        "是否中文": d.get("是否中文", True),
        "中文譯文": _s(d.get("中文譯文")),
        "意圖分類方式": d.get("意圖分類方式"),
        "負責機關": _s(d.get("負責機關")),
        "負責單位": _s(d.get("負責單位")),
        "命中術語": d.get("命中術語"),
        "enriched_query": _s(d.get("enriched_query")),
        "similarity_threshold": d.get("similarity_threshold"),
        "top_n": d.get("top_n"),
        "問卷命中數": d.get("問卷命中數"),
        "知識命中數": d.get("知識命中數"),
        "參考來源": d.get("參考來源"),
        "知識參考來源": d.get("知識參考來源"),
        "raw_context": _s(d.get("raw_context")),
        "knowledge_context": _s(d.get("knowledge_context")),
        "dictionary_xml": _s(d.get("dictionary_xml")),
        "llm_prompt": _s(d.get("llm_prompt")),
        "llm_output": _s(d.get("llm_output")),
        "英文回覆": _s(d.get("英文回覆")),
        "中文回覆": _s(d.get("中文回覆")),
        "耗時毫秒": d.get("耗時毫秒"),
        "錯誤訊息": _s(d.get("錯誤訊息")),
    }
