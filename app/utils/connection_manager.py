"""
connection_manager.py — 非同步連線監控上下文管理器與可中斷任務執行器

公開介面：
    monitor_disconnect(request, check_interval) — 上下文管理器，監聽客戶端斷線並
        產出 interrupt_signal (asyncio.Event)，離開時保證背景任務完整終止。

    run_interruptible(coro, interrupt_signal, label) — 可中斷包裝器，讓任意
        coroutine 在 interrupt_signal 觸發時立即取消，適用於無內建中斷機制的 I/O
        操作（DB 查詢、Embedding、CPU-bound thread 等）。

使用範例：
    async with monitor_disconnect(request) as interrupt_signal:
        retrieval_result = await run_interruptible(
            hybrid_retrieve(query, session),
            interrupt_signal,
            label="Phase 2 向量檢索",
        )
        llm_output = await chat_completion(prompt, interrupt_event=interrupt_signal)
"""
import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Coroutine
from typing import Any, TypeVar

from fastapi import Request

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ──────────────────────────────────────────────────────────────────────────────
# 1. 斷線監聽上下文管理器
# ──────────────────────────────────────────────────────────────────────────────

@contextlib.asynccontextmanager
async def monitor_disconnect(
    request: Request,
    check_interval: float = 0.5,
) -> AsyncGenerator[asyncio.Event, None]:
    """
    非同步上下文管理器：監聽 HTTP 客戶端連線狀態。

    在背景啟動輪詢任務，偵測到客戶端斷線時觸發 ``interrupt_signal``。
    無論業務邏輯正常結束或拋出例外，均保證背景任務被完整終止，防止記憶體洩漏。

    Args:
        request: FastAPI Request 物件，用於呼叫 ``is_disconnected()``。
        check_interval: 輪詢間隔秒數（預設 0.5 秒）。

    Yields:
        interrupt_signal (asyncio.Event): 中斷訊號。已觸發時（``is_set()`` 為 True）
            表示客戶端已斷線，底層操作應提早結束。

    Example::

        async with monitor_disconnect(request) as interrupt_signal:
            result = await run_interruptible(some_coro(), interrupt_signal)
    """
    interrupt_signal = asyncio.Event()

    async def _watch_disconnect() -> None:
        """背景監聽：等待 ASGI 的 http.disconnect 訊息，斷線即設置訊號。"""
        try:
            while True:
                message = await request.receive()
                if message.get("type") == "http.disconnect":
                    logger.info(
                        "偵測到客戶端斷線（路徑: %s），觸發中斷訊號",
                        request.url.path,
                    )
                    interrupt_signal.set()
                    break
        except asyncio.CancelledError:
            # 由 finally 區塊主動取消，屬於預期內行為，靜默處理不記錄錯誤
            pass

    watcher = asyncio.create_task(_watch_disconnect())

    try:
        yield interrupt_signal
    finally:
        # 強制終止背景任務（無論業務邏輯是否正常結束）
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass  # 預期內的取消，靜默吸收
        logger.debug(
            "monitor_disconnect 背景任務已完整終止（路徑: %s）",
            request.url.path,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. 可中斷任務包裝器
# ──────────────────────────────────────────────────────────────────────────────

async def run_interruptible(
    coro: Coroutine[Any, Any, T],
    interrupt_signal: asyncio.Event,
    label: str = "",
) -> T:
    """
    執行 coroutine，若 ``interrupt_signal`` 在執行期間觸發，立即取消並拋出
    ``asyncio.CancelledError``。適用於本身無中斷機制的 I/O 密集或長耗時操作。

    實作採用「雙任務競速」策略：
    - ``task``：執行目標 coroutine。
    - ``_signal_watcher``：等待 interrupt_signal，觸發後取消 task。
    兩者均在 finally 中保證清理，避免 orphan task。

    Args:
        coro: 目標 coroutine（尚未 await 的 coroutine 物件）。
        interrupt_signal: 由 ``monitor_disconnect`` 產出的中斷事件。
        label: 操作名稱，用於日誌識別（建議格式：「Phase X 操作說明」）。

    Returns:
        coroutine 的回傳值。

    Raises:
        asyncio.CancelledError: 客戶端斷線，操作被提前取消時拋出。
        其他例外: 原 coroutine 本身拋出的任何例外照常傳遞。

    Example::

        retrieval = await run_interruptible(
            hybrid_retrieve(query, session),
            interrupt_signal,
            label="Phase 2 向量檢索",
        )
    """
    # 快速路徑：進入前已斷線，直接取消，省略不必要的任務建立
    if interrupt_signal.is_set():
        coro.close()  # 釋放尚未啟動的 coroutine 資源
        logger.info("操作在啟動前即被中斷，跳過執行：%s", label or "(unnamed)")
        raise asyncio.CancelledError(f"client disconnected before: {label}")

    task: asyncio.Task[T] = asyncio.ensure_future(coro)

    async def _signal_watcher() -> None:
        """等待中斷訊號，觸發後取消主任務。"""
        await interrupt_signal.wait()
        if not task.done():
            task.cancel()

    watcher = asyncio.ensure_future(_signal_watcher())

    try:
        return await task
    except asyncio.CancelledError:
        logger.info(
            "操作因客戶端斷線而取消（%s）",
            label or "(unnamed)",
        )
        raise
    finally:
        # 確保 _signal_watcher 不殘留
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
