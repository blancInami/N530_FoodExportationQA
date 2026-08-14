"""
LLM service: calls the remote LLM server for text generation.
"""
import logging
import time

import httpx
from fastapi import HTTPException

from app.config import get_settings

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = "You are a professional food export regulation assistant."


# async def chat_completion(
#     prompt: str,
#     temperature: float = 0,
#     max_tokens: int = 4096,
#     system_prompt: str | None = None,
# ) -> str:
#     """
#     Call the LLM server (OpenAI-compatible /v1/chat/completions).
#     Returns the generated text content.

#     Args:
#         prompt: User message content.
#         temperature: Sampling temperature (0 = deterministic).
#         max_tokens: Maximum tokens for the response.
#         system_prompt: Optional system message. Defaults to the food export assistant prompt.
#     """
#     settings = get_settings()
#     # Gateway mode: URL is used as-is; otherwise append OpenAI-compatible path
#     if settings.use_gateway:
#         url = settings.llm_url
#     else:
#         url = f"{settings.llm_url}/v1/chat/completions"
#     headers = settings.gw_headers
#     payload = {
#         "model": settings.llm_model,
#         "messages": [
#             {"role": "system", "content": system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT},
#             {"role": "user", "content": prompt},
#         ],
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#     }

#     logger.info(
#         "LLM 請求：url=%s  提示長度=%d  temperature=%s  max_tokens=%d",
#         url, len(prompt), temperature, max_tokens,
#     )
#     logger.debug("LLM 提示預覽：%s ...", prompt[:200])
#     start = time.perf_counter()

#     async with httpx.AsyncClient(timeout=120.0) as client:
#         try:
#             resp = await client.post(url, json=payload, headers=headers)
#             resp.raise_for_status()
#         except httpx.HTTPStatusError as e:
#             logger.error("LLM 伺服器 HTTP 錯誤：狀態碼=%s  url=%s", e.response.status_code, url)
#             raise HTTPException(status_code=502, detail=f"LLM server returned {e.response.status_code}")
#         except httpx.RequestError as e:
#             logger.error("LLM 伺服器無法連線：%s  url=%s", e, url)
#             raise HTTPException(status_code=503, detail=f"LLM server unreachable: {e}")

#     elapsed_ms = (time.perf_counter() - start) * 1000
#     data = resp.json()
#     raw_content = data["choices"][0]["message"].get("content")
#     if raw_content is None:
#         logger.warning("LLM 回傳 content=null（可能觸發了 tool call 或回應被截斷），以空字串替代")
#         raw_content = ""
#     content: str = raw_content
#     logger.info("LLM 回應完成：輸出長度=%d  耗時=%.1f ms", len(content), elapsed_ms)
#     logger.debug("LLM 回應預覽：%s ...", content[:200])
#     return content

import asyncio
import json
async def chat_completion(
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 4096,
    system_prompt: str | None = None,
    interrupt_event: asyncio.Event | None = None,  # 新增：用於控制中斷的異步事件
) -> str:
    """
    Call the LLM server (OpenAI-compatible /v1/chat/completions) with streaming for resource optimization.
    Returns the accumulated text content even if interrupted early.
    """
    settings = get_settings()
    if settings.use_gateway:
        url = settings.llm_url
    else:
        url = f"{settings.llm_url}/v1/chat/completions"
    headers = settings.gw_headers
    
    # 關鍵修改 1：強制開啟 stream 模式
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,  
    }

    logger.info(
        "LLM 串流請求啟動：url=%s  提示長度=%d  temperature=%s  max_tokens=%d",
        url, len(prompt), temperature, max_tokens,
    )
    start = time.perf_counter()
    
    # 用於在串流過程中累積文字片段
    chunks = []

    # 關鍵修改 2：將超時時間切分為連線(connect)與讀取(read)，並使用 client.stream 上下文管理器
    # 移除全域大超時，改由串流迭代控制
    timeout = httpx.Timeout(120.0, connect=5.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                
                # 關鍵修改 3：逐行讀取 SSE (Server-Sent Events) 串流
                async for line in resp.aiter_lines():
                    
                    # 檢查點：若外部偵測到需要中斷（如前端斷線或業務取消）
                    if interrupt_event is not None and interrupt_event.is_set():
                        logger.warning("檢測到外部中斷訊號，主動關閉連線以空出 LLM 資源")
                        raise asyncio.CancelledError("Client disconnected")
                    
                    if not line.strip():
                        continue
                        
                    if line.startswith("data: "):
                        clean_line = line[6:].strip()
                        
                        if clean_line == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(clean_line)
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            content_chunk = delta.get("content")
                            
                            if content_chunk:
                                chunks.append(content_chunk)
                                
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            logger.debug("解析流式數據行失敗 (可能為填充資訊): %s, 錯誤: %s", line, e)
                            continue

        except httpx.HTTPStatusError as e:
            logger.error("LLM 伺服器 HTTP 錯誤：狀態碼=%s  url=%s", e.response.status_code, url)
            raise HTTPException(status_code=502, detail=f"LLM server returned {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error("LLM 伺服器無法連線：%s  url=%s", e, url)
            raise HTTPException(status_code=503, detail=f"LLM server unreachable: {e}")
        except asyncio.CancelledError:
            # 處理整個 FastAPI Task 被協程框架取消的情境（例如客戶端斷開 ASGI 連線）
            logger.info("FastAPI Task 協程被取消，已釋放連線資源，拋出例外以中斷管線流程。")
            raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    
    # 關鍵修改 4：組裝已生成的內容
    content = "".join(chunks)
    logger.info("LLM 處理完成（或部分中斷）：輸出總長度=%d  耗時=%.1f ms", len(content), elapsed_ms)
    logger.debug("LLM 回應預覽：%s ...", content[:200])
    return content


async def chat_completion_vision(
    prompt: str,
    images: list[dict[str, str]],  # List of {"content_b64": str, "mime_type": str}
    temperature: float = 0,
    max_tokens: int = 4096,
    system_prompt: str | None = None,
    interrupt_event: asyncio.Event | None = None,
) -> str:
    """
    Call the VLM / multimodal LLM endpoint with visual content (OpenAI-compatible format).
    
    Args:
        prompt: User text instructions.
        images: List of image objects with base64 strings and MIME types.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        system_prompt: System prompt instructions.
        interrupt_event: Async event to signal cancellation.
    """
    settings = get_settings()
    
    # 決定 URL 與模型名稱（優先使用 vlm 設定，未填則回退至主 llm 設定）
    model_name = settings.vlm_model or settings.llm_model
    if settings.vlm_url:
        url = settings.vlm_url if settings.use_gateway else f"{settings.vlm_url}/v1/chat/completions"
    elif settings.use_gateway:
        url = settings.llm_url
    else:
        url = f"{settings.llm_url}/v1/chat/completions"
        
    headers = settings.gw_headers

    # 組裝多模態 user content
    user_content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for img in images:
        b64 = img.get("content_b64", "")
        mime = img.get("mime_type", "image/jpeg")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
            },
        })

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    logger.info(
        "VLM 串流請求啟動：url=%s  model=%s  圖片數=%d  提示長度=%d",
        url, model_name, len(images), len(prompt),
    )
    start = time.perf_counter()
    chunks = []
    timeout = httpx.Timeout(180.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if interrupt_event is not None and interrupt_event.is_set():
                        logger.warning("檢測到外部中斷訊號，主動關閉連線以空出 VLM 資源")
                        raise asyncio.CancelledError("Client disconnected")

                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        clean_line = line[6:].strip()
                        if clean_line == "[DONE]":
                            break
                        try:
                            data = json.loads(clean_line)
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            content_chunk = delta.get("content")
                            if content_chunk:
                                chunks.append(content_chunk)
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            logger.debug("解析流式數據行失敗: %s, 錯誤: %s", line, e)
                            continue

        except httpx.HTTPStatusError as e:
            logger.error("VLM 伺服器 HTTP 錯誤：狀態碼=%s  url=%s", e.response.status_code, url)
            raise HTTPException(status_code=502, detail=f"VLM server returned {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error("VLM 伺服器無法連線：%s  url=%s", e, url)
            raise HTTPException(status_code=503, detail=f"VLM server unreachable: {e}")
        except asyncio.CancelledError:
            logger.info("FastAPI Task 協程被取消，釋放 VLM 連線資源。")
            raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    content = "".join(chunks)
    logger.info("VLM 處理完成：輸出總長度=%d  耗時=%.1f ms", len(content), elapsed_ms)
    return content