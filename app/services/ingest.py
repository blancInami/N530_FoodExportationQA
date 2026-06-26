"""
Ingest service: chunking, embedding, and batch writing to 問卷題目切塊.
"""
import logging
import uuid

import tiktoken
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import 問卷題目切塊, 問卷題目檔
from app.services.embedding import get_embeddings_batch
from app.utils import sanitize_text_for_db, sliding_window_chunk

logger = logging.getLogger(__name__)

_encoder = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoder.encode(text))


async def ingest_questionnaire(questionnaire_pk: str, session: AsyncSession) -> dict:
    """
    Ingest (chunk + vectorize) all question items for a given 問卷主檔主鍵.
    """
    settings = get_settings()
    logger.info("開始匯入問卷：問卷主檔主鍵=%s", questionnaire_pk)

    # Step 1: Fetch question items
    stmt = (
        select(
            問卷題目檔.c["主鍵"],
            問卷題目檔.c["題目"],
            問卷題目檔.c["回覆"],
        )
        .where(問卷題目檔.c["問卷主檔主鍵"] == questionnaire_pk)
        .where(
            (問卷題目檔.c["是否刪除"].is_(None)) | (問卷題目檔.c["是否刪除"] == 0)
        )
        .order_by(問卷題目檔.c["排序"])
    )
    result = await session.execute(stmt)
    items = result.mappings().all()

    if not items:
        logger.warning("匯入問卷：找不到有效題目，問卷主檔主鍵=%s", questionnaire_pk)
        return {"status": "no_items", "message": "No active question items found."}

    logger.info("匯入問卷：共 %d 筆題目待處理", len(items))
    total_chunks_written = 0

    for item_idx, item in enumerate(items, start=1):
        item_pk = item["主鍵"]
        title_text = item["題目"] or ""
        reply_text = item["回覆"] or ""

        # Step 2: Chunk with sliding window — 題目與回覆各自獨立切塊，不對齊長度
        title_chunks = [
            sanitize_text_for_db(c)
            for c in sliding_window_chunk(
                title_text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
            )
        ]
        reply_chunks = [
            sanitize_text_for_db(c)
            for c in sliding_window_chunk(
                reply_text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
            )
        ]

        logger.debug(
            "匯入題目 [%d/%d]：主鍵=%s  question 切塊=%d  answer 切塊=%d",
            item_idx, len(items), item_pk, len(title_chunks), len(reply_chunks),
        )

        # Step 3: 合併所有 chunk_text，一次呼叫 get_embeddings_batch
        all_texts = title_chunks + reply_chunks
        all_texts_for_embed = [t if t else "empty" for t in all_texts]

        logger.info(
            "匯入題目 [%d/%d]：批次嵌入 — 總切塊=%d（question=%d  answer=%d）",
            item_idx, len(items), len(all_texts_for_embed),
            len(title_chunks), len(reply_chunks),
        )
        all_vectors = await get_embeddings_batch(all_texts_for_embed)

        # Step 4: Build records list
        records: list[dict] = []
        for idx, (chunk_text, vector) in enumerate(
            zip(all_texts, all_vectors[:len(title_chunks)])
        ):
            records.append({
                "主鍵": uuid.uuid4(),
                "問卷題目檔主鍵": item_pk,
                "chunk_source": "question",
                "chunk_text": chunk_text,
                "embedding": vector,
                "切塊索引": idx,
                "詞元數量": _count_tokens(chunk_text),
            })
        for idx, (chunk_text, vector) in enumerate(
            zip(reply_chunks, all_vectors[len(title_chunks):])
        ):
            records.append({
                "主鍵": uuid.uuid4(),
                "問卷題目檔主鍵": item_pk,
                "chunk_source": "answer",
                "chunk_text": chunk_text,
                "embedding": vector,
                "切塊索引": idx,
                "詞元數量": _count_tokens(chunk_text),
            })

        # Step 5: DELETE old chunks, INSERT new ones
        await session.execute(
            delete(問卷題目切塊).where(問卷題目切塊.c["問卷題目檔主鍵"] == item_pk)
        )
        if records:
            await session.execute(insert(問卷題目切塊), records)

        logger.info(
            "匯入題目 [%d/%d]：主鍵=%s  刪除舊切塊並寫入 %d 筆新切塊",
            item_idx, len(items), item_pk, len(records),
        )
        total_chunks_written += len(records)

    await session.commit()

    logger.info(
        "問卷匯入完成：問卷主檔主鍵=%s  題目筆數=%d  總切塊數=%d",
        questionnaire_pk, len(items), total_chunks_written,
    )
    return {
        "status": "success",
        "questionnaire_pk": questionnaire_pk,
        "items_processed": len(items),
        "total_chunks_written": total_chunks_written,
    }