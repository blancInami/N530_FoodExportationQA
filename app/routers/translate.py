"""
Translation Router: /api/v1/translate
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db_session
from app.schemas.translate import TranslateRequest, TranslateResponse
from app.services.translate import translate_with_dictionary
from app.utils.connection_manager import monitor_disconnect

router = APIRouter(prefix="/api/v1/translation", tags=["Translation"])
logger = logging.getLogger(__name__)


@router.post(
    "/re-translate",
    response_model=TranslateResponse,
    summary="中文術語約束翻譯",
    description=(
        "將中文文本翻譯為英文，翻譯過程中自動比對「官方正規詞彙」資料庫（6000+ 筆中英對照術語），"
        "確保農業、食品輸銷相關專業術語以官方核定英文名詞輸出。\n\n"
        "**處理流程：**\n\n"
        "1. 以 Aho-Corasick DictionaryMatcher 在輸入文本中快速比對官方術語。\n"
        "2. 命中術語以 `{中文術語} → {英文術語}` 格式注入翻譯 prompt，強制 LLM 使用官方譯名。\n"
        "3. 呼叫 LLM（temperature=0）生成確定性英文翻譯結果。\n"
        "4. 回傳原始中文文本與翻譯後英文文本。\n\n"
        "**注意事項：**\n"
        "- 輸入文本長度限制：1～5000 字元。\n"
        "- 本端點專為「官方術語標準化輸出」設計，一般翻譯需求建議直接使用 `/ask` 端點。"
    ),
    response_description="回傳原始輸入文本（`original`）與術語約束翻譯結果（`translation`）。",
    responses={
        200: {"description": "翻譯成功"},
        422: {"description": "請求格式驗證失敗（如文本超過 5000 字元）"},
        500: {"description": "系統內部錯誤（LLM 或資料庫異常）"},
    },
)
async def translate_text(
    body: TranslateRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> TranslateResponse:
    logger.info("翻譯請求：輸入長度=%d", len(body.text))
    async with monitor_disconnect(request) as interrupt_signal:
        translation = await translate_with_dictionary(body.text, session, interrupt_event=interrupt_signal)
    logger.info("翻譯完成：輸出長度=%d", len(translation))
    return TranslateResponse(original=body.text, translation=translation)
