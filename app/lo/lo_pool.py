"""LibreOffice 長駐實例資源池（Daemon Worker Pool）。

架構概覽
--------
- LibreOfficeWorker：封裝單一 soffice.exe daemon 的生命週期與轉檔操作。
- LibreOfficePool  ：以 asyncio.Queue 管理多個 Worker；非同步背景預熱，
                     轉檔失敗時自動重啟壞 Worker 後歸還佇列。

預熱流程（服務先啟動）
---------------------
lifespan 以 asyncio.create_task(pool.initialize()) 非同步預熱，
FastAPI 不阻塞 — 服務立即可接受請求；Worker 就緒後才進入 queue，
首批請求若在任一 Worker 就緒前到達則自然 block waiting。

資源回收
--------
每個 Worker 帶 conversion_count；達 LO_MAX_CONVERSIONS 上限後，
release() 自動執行 stop() → reset → start() 再放回 queue。
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import socket
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from app.core.config import Settings

logger = logging.getLogger(__name__)

# LibreOffice Portable 路徑（相對於此模組上兩層目錄）
_LO_PROGRAM_DIR: Path = (
    Path(__file__).parent.parent.parent / "tools" / "LibreOfficePortable" / "App"
    / "libreoffice" / "program"
)
_SOFFICE_EXE: Path = _LO_PROGRAM_DIR / "soffice.exe"
_LO_PYTHON_EXE: Path = _LO_PROGRAM_DIR / "python.exe"
_CONVERT_SCRIPT: Path = Path(__file__).parent / "lo_convert_script.py"

# TCP 就緒輪詢設定
_READY_POLL_INTERVAL: float = 0.5   # 秒
_READY_TIMEOUT: float = 60.0        # 秒


async def _wait_for_port(host: str, port: int, timeout: float) -> None:
    """輪詢直到 TCP port 可連線，或逾時後 raise RuntimeError。"""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=2.0
            )
            writer.close()
            await writer.wait_closed()
            return
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise RuntimeError(
                    f"LibreOffice daemon 啟動逾時（{timeout}s），port={port}"
                )
            await asyncio.sleep(_READY_POLL_INTERVAL)


class LibreOfficeWorker:
    """封裝單一 LibreOffice daemon 實例的生命週期。"""

    def __init__(self, worker_id: int, base_port: int) -> None:
        self.worker_id: int = worker_id
        self.port: int = base_port + worker_id + 1   # base=2000 → 2001, 2002, …
        self.conversion_count: int = 0
        self._process: asyncio.subprocess.Process | None = None
        self._profile_root: Path | None = None   # tempdir，每次 start() 重建

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """啟動 soffice.exe daemon 並等待 TCP port 就緒。"""
        # 每次 start() 建立新的獨立暫存 UserInstallation（避免 profile 鎖定）
        profile_root = Path(tempfile.mkdtemp(prefix=f"lo_worker_{self.worker_id}_"))
        self._profile_root = profile_root
        profile_uri = profile_root.as_uri()

        cmd = [
            str(_SOFFICE_EXE),
            f"-env:UserInstallation={profile_uri}",
            "--headless",
            "--invisible",
            "--nodefault",
            "--norestore",
            "--nofirststartwizard",
            "--nolockcheck",
            f'--accept=socket,host=127.0.0.1,port={self.port};urp;',
        ]

        logger.info(
            "[lo_pool] worker_%d 啟動 soffice.exe port=%d profile=%s",
            self.worker_id, self.port, profile_root,
        )

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            await _wait_for_port("127.0.0.1", self.port, _READY_TIMEOUT)
        except RuntimeError:
            # 啟動失敗：清理後重新拋出
            await self.stop()
            raise

        logger.info(
            "[lo_pool] worker_%d 就緒 port=%d pid=%s",
            self.worker_id, self.port, self._process.pid,
        )

    async def convert(self, docx_path: Path, output_dir: Path) -> Path:
        """透過 UNO 橋接將 DOCX 轉為 PDF；成功後遞增計數器。

        Args:
            docx_path:  輸入的 DOCX 絕對路徑。
            output_dir: 輸出目錄（PDF 將以與 docx_path 同名的 .pdf 命名）。

        Returns:
            輸出 PDF 的絕對路徑。

        Raises:
            RuntimeError: 轉換失敗（包含 UNO 連線失敗、LO 拋錯等）。
        """
        output_pdf = output_dir / (docx_path.stem + ".pdf")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            "[lo_pool] worker_%d 開始轉換 %s port=%d",
            self.worker_id, docx_path.name, self.port,
        )

        proc = await asyncio.create_subprocess_exec(
            str(_LO_PYTHON_EXE),
            str(_CONVERT_SCRIPT),
            str(self.port),
            str(docx_path.resolve()),
            str(output_pdf.resolve()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        stdout = stdout_b.decode("utf-8", errors="replace").strip()
        stderr = stderr_b.decode("utf-8", errors="replace").strip()

        if stdout:
            logger.debug("[lo_pool] worker_%d UNO stdout: %s", self.worker_id, stdout)
        if stderr:
            logger.warning("[lo_pool] worker_%d UNO stderr: %s", self.worker_id, stderr)

        if proc.returncode != 0:
            raise RuntimeError(
                f"worker_{self.worker_id} UNO 轉換失敗 "
                f"returncode={proc.returncode} stderr={stderr!r}"
            )

        if not output_pdf.exists():
            raise RuntimeError(
                f"worker_{self.worker_id} UNO 轉換後找不到 PDF：{output_pdf}"
            )

        self.conversion_count += 1
        logger.info(
            "[lo_pool] worker_%d 轉換完成 count=%d file=%s",
            self.worker_id, self.conversion_count, output_pdf.name,
        )
        return output_pdf

    async def stop(self) -> None:
        """安全終止 soffice.exe 子行程並清理 profile 目錄。"""
        if self._process is not None:
            proc = self._process
            self._process = None
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except ProcessLookupError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "[lo_pool] worker_%d terminate 逾時，強制 kill pid=%s",
                    self.worker_id, proc.pid,
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

        if self._profile_root is not None:
            profile = self._profile_root
            self._profile_root = None
            # Windows 在行程終止後可能短暫保留檔案控制碼，
            # 加入重試進行清除（最多 5 次，每次等待 0.5 s）
            for attempt in range(5):
                try:
                    shutil.rmtree(profile)
                    logger.debug(
                        "[lo_pool] worker_%d 暂存目錄已清除：%s", self.worker_id, profile
                    )
                    break
                except OSError:
                    if attempt < 4:
                        await asyncio.sleep(0.5)
                    else:
                        logger.warning(
                            "[lo_pool] worker_%d 無法清除暂存目錄，請手動刪除：%s",
                            self.worker_id, profile,
                        )

        logger.info("[lo_pool] worker_%d 已停止", self.worker_id)


# ── Pool ──────────────────────────────────────────────────────────────────────


class LibreOfficePool:
    """以 asyncio.Queue 管理多個 LibreOfficeWorker 的資源池。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        size = max(0, settings.LO_POOL_SIZE)
        self._queue: asyncio.Queue[LibreOfficeWorker] = asyncio.Queue(maxsize=size)
        self._workers: list[LibreOfficeWorker] = [
            LibreOfficeWorker(i, settings.LO_BASE_PORT) for i in range(size)
        ]
        self._init_task: asyncio.Task[None] | None = None  # 背景預熱 task，用於 shutdown 時取消

    # ── initialization ───────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """預熱所有 Worker：每個 Worker 就緒後才推入 queue。

        可直接 await（LO_POOL_EAGER=true）或透過 start_background() 轉化為 task。
        """
        for worker in self._workers:
            try:
                await worker.start()
                await self._queue.put(worker)
                logger.info(
                    "[lo_pool] worker_%d 已加入資源池（%d/%d 就緒）",
                    worker.worker_id,
                    self._queue.qsize(),
                    len(self._workers),
                )
            except asyncio.CancelledError:
                logger.info("[lo_pool] 預熱 task 被取消，停止啟動剩餘 Worker")
                raise
            except Exception:
                logger.exception(
                    "[lo_pool] worker_%d 啟動失敗，跳過此實例", worker.worker_id
                )

    def start_background(self) -> asyncio.Task[None]:
        """建立背景預熱 task 並回傳它（用於非連續就緒模式）。

        返回的 task 引用由 shutdown() 內部儲存，對外無需自行管理。
        """
        self._init_task = asyncio.create_task(self.initialize())
        return self._init_task

    # ── acquire / release ─────────────────────────────────────────────────────

    async def acquire(self) -> LibreOfficeWorker:
        """從佇列中取出一個可用的 Worker（無閒置資源時自動 block waiting）。"""
        return await self._queue.get()

    async def release(self, worker: LibreOfficeWorker) -> None:
        """歸還 Worker；達到轉檔上限時先回收重啟再放回。"""
        if worker.conversion_count >= self._settings.LO_MAX_CONVERSIONS:
            logger.info(
                "[lo_pool] worker_%d 達到轉檔上限（%d），執行回收重啟",
                worker.worker_id, worker.conversion_count,
            )
            await worker.stop()
            worker.conversion_count = 0
            try:
                await worker.start()
            except Exception:
                logger.exception(
                    "[lo_pool] worker_%d 回收後重啟失敗，從池中移除", worker.worker_id
                )
                return   # 不放回 queue：池縮減
        await self._queue.put(worker)

    @asynccontextmanager
    async def acquire_context(self) -> AsyncIterator[LibreOfficeWorker]:
        """context manager：自動取得與歸還 Worker；轉檔失敗時重啟後歸還。

        使用方式::

            async with pool.acquire_context() as worker:
                pdf = await worker.convert(docx, out_dir)
        """
        worker = await self.acquire()
        failed = False
        try:
            yield worker
        except Exception:
            failed = True
            raise
        finally:
            if failed:
                # 轉換失敗：重啟壞 Worker 再歸還，確保 queue 不收到不健康的實例
                logger.warning(
                    "[lo_pool] worker_%d 轉換過程拋出例外，執行重啟後歸還",
                    worker.worker_id,
                )
                await worker.stop()
                worker.conversion_count = 0
                try:
                    await worker.start()
                    await self._queue.put(worker)
                except Exception:
                    logger.exception(
                        "[lo_pool] worker_%d 異常後重啟失敗，從池中永久移除",
                        worker.worker_id,
                    )
            else:
                await self.release(worker)

    # ── shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """關閉所有 Worker（FastAPI shutdown 時呼叫）。"""
        # 1. 如果背景預熱 task 尚在執行，先取消它（避免新 Worker 在 stop 之後被啟動）
        if self._init_task is not None and not self._init_task.done():
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                pass
            logger.info("[lo_pool] 預熱 task 已取消")

        # 2. 逐一停止所有 Worker（包含未入佇列的實例）
        logger.info("[lo_pool] 開始關閉所有 LibreOffice Worker（共 %d 個）", len(self._workers))
        for worker in self._workers:
            try:
                await worker.stop()
            except Exception:
                logger.exception("[lo_pool] worker_%d shutdown 時發生錯誤", worker.worker_id)
        logger.info("[lo_pool] 所有 Worker 已關閉")
