"""
Embedding service: calls the remote embedding server to produce vector(1024).
"""
import logging
import time

import httpx
from fastapi import HTTPException

from app.config import get_settings

logger = logging.getLogger(__name__)


async def get_embedding(text: str) -> list[float]:
    """
    Call the embedding server to get a 1024-dim vector for the given text.
    """
    settings = get_settings()
    # Gateway mode: URL is used as-is; otherwise append OpenAI-compatible path
    if settings.use_gateway:
        url = settings.embedding_url
    else:
        url = f"{settings.embedding_url}/v1/embeddings"
    headers = settings.gw_headers
    payload = {"model": settings.embedding_model, "input": text}

    logger.debug("單筆嵌入向量：文字長度=%d  url=%s", len(text), url)
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("嵌入伺服器 HTTP 錯誤：狀態碼=%s  url=%s", e.response.status_code, url)
            raise HTTPException(status_code=502, detail=f"Embedding server returned {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error("嵌入伺服器無法連線：%s  url=%s", e, url)
            raise HTTPException(status_code=503, detail=f"Embedding server unreachable: {e}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = resp.json()
    embedding = data["data"][0]["embedding"]
    logger.info("單筆嵌入完成：維度=%d  耗時=%.1f ms", len(embedding), elapsed_ms)
    return embedding


async def get_embeddings_batch(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """
    Batch embedding for multiple texts with chunking to avoid 413 Payload Too Large.
    """
    if not texts:
        return []

    settings = get_settings()
    # Gateway mode: URL is used as-is; otherwise append OpenAI-compatible path
    if settings.use_gateway:
        url = settings.embedding_url
    else:
        url = f"{settings.embedding_url}/v1/embeddings"
    headers = settings.gw_headers
    all_results = []

    logger.debug("批次嵌入向量啟動：總筆數=%d  每批大小=%d  url=%s", len(texts), batch_size, url)
    start_total = time.perf_counter()

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            payload = {"model": settings.embedding_model, "input": batch_texts}

            try:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error("嵌入伺服器 HTTP 錯誤（批次）：狀態碼=%s  url=%s", e.response.status_code, url)
                raise HTTPException(status_code=502, detail=f"Embedding server returned {e.response.status_code}")
            except httpx.RequestError as e:
                logger.error("嵌入伺服器無法連線（批次）：%s  url=%s", e, url)
                raise HTTPException(status_code=503, detail=f"Embedding server unreachable: {e}")

            data = resp.json()
            # Remote server might return unsorted data or different indexing
            sorted_batch_data = sorted(data["data"], key=lambda x: x["index"])
            all_results.extend([item["embedding"] for item in sorted_batch_data])

    elapsed_ms = (time.perf_counter() - start_total) * 1000
    dim = len(all_results[0]) if all_results else 0
    logger.info("批次嵌入完成：總筆數=%d  維度=%d  總耗時=%.1f ms", len(all_results), dim, elapsed_ms)
    return all_results