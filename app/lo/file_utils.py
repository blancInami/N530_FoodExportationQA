from __future__ import annotations

import asyncio
import base64
import io
import logging
import subprocess
import tempfile
import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING
import shutil
if TYPE_CHECKING:
    from app.lo.lo_pool import LibreOfficeWorker
from app.config import Settings, get_settings
logger = logging.getLogger(__name__)

# Poppler bin 路徑（由此模組位置計算，不受 CWD 影響）
POPPLER_BIN_PATH: Path = Path(__file__).parent.parent.parent / "tools"  / "poppler" / "bin"

# LibreOffice Portable soffice.exe 路徑
SOFFICE_PATH: Path =  Path(__file__).parent.parent.parent / "tools" / "LibreOfficePortable" / "App" / "libreoffice" / "program" / "soffice.exe"


_PDF_MIME = "application/pdf"

# Office 格式副檔名集合（DOCX / DOC / ODT / RTF / XLS / XLSX / ODS / PPT / PPTX / ODP）
# 這些格式統一先透過 LibreOffice 轉為 PDF，再逐頁轉 JPEG 餵入 VLM
_OFFICE_EXTS: frozenset[str] = frozenset({
    ".doc", ".docx", ".odt", ".rtf",
    ".xls", ".xlsx", ".ods",
    ".ppt", ".pptx", ".odp",
})

# 圖片格式副檔名 → 正規化後的 MIME 類型
# mimetypes.guess_type 對 .jpg 等有時會回傳 None 或不正確的值，統一在此查表
_IMAGE_EXT_TO_MIME: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
}


def _docx_to_pdf_via_libreoffice(docx_path: Path, output_dir: Path, timeout_seconds: int = 60) -> Path:
    """同步：使用 LibreOffice Portable soffice.exe 將 DOCX 轉為 PDF（於 executor 中呼叫）。

    LibreOffice 會在 output_dir 下產生與 docx_path 同名、副檔名為 .pdf 的檔案。
    """
    if not SOFFICE_PATH.exists():
        raise FileNotFoundError(f"找不到 LibreOffice 執行檔：{SOFFICE_PATH}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用 uuid 確保 profile 命名的絕對唯一性，避免高併發下的目錄衝突
    import uuid
    profile_name = f"lo_profile_{uuid.uuid4().hex}"
    lo_profile = output_dir / profile_name
    lo_profile.mkdir(exist_ok=True)

    try:
        process = subprocess.run(
            [
                str(SOFFICE_PATH),
                f"-env:UserInstallation=file:///{lo_profile.as_posix()}",
                "--headless",
                "--invisible",
                "--nodefault",
                "--norestore",
                "--convert-to", "pdf",
                "--outdir", str(output_dir),
                str(docx_path),
            ],
            capture_output=True,
            timeout=timeout_seconds, # 強制設置時間邊界
        )
        
        stdout = process.stdout.decode("utf-8", errors="ignore")
        stderr = process.stderr.decode("utf-8", errors="ignore")
        
        if process.returncode != 0:
            raise RuntimeError(f"LibreOffice 轉換失敗（code {process.returncode}）：{stderr}")
            
        pdf_path = output_dir / (docx_path.stem + ".pdf")
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"轉換後找不到預期的 PDF：{pdf_path}\nstdout: {stdout}\nstderr: {stderr}"
            )
            
        logger.info("[file_utils] LibreOffice DOCX->PDF 成功：%s", pdf_path)
        return pdf_path
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"[file_utils] LibreOffice 轉換逾時 (> {timeout_seconds}s)：{docx_path}")
        raise RuntimeError(f"文件轉換超時，已強制終止作業。") from e
    finally:
        # 無論成功或失敗，強制回收獨立的 User Profile 資源
        if lo_profile.exists():
            shutil.rmtree(lo_profile, ignore_errors=True)


def _pdf_to_image_pages(pdf_path: Path, dpi: int | None = None) -> list[dict[str, str]]:
    """同步：將 PDF 的每一頁轉為 JPEG base64 dict（於 executor 中呼叫）。"""
    from pdf2image import convert_from_path  # type: ignore[import]
    from app.config import get_settings

    settings = get_settings()
    effective_dpi = dpi if dpi is not None else (settings.vlm_dpi or 150)

    pages = convert_from_path(
        str(pdf_path),
        dpi=effective_dpi,
        poppler_path=str(POPPLER_BIN_PATH),
    )
    result: list[dict[str, str]] = []
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="JPEG", quality=80, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        result.append({"content_b64": b64, "mime_type": "image/jpeg"})
    return result


def _try_load_cached_images(file_md5: str, settings: Settings) -> list[dict[str, str]] | None:
    """若啟用 vlm_save_temp_images 且 temp_images/{file_md5}/ 存在完整圖片，直接讀取快取。"""
    if not getattr(settings, "vlm_save_temp_images", False):
        return None
    cache_dir = Path(getattr(settings, "vlm_temp_images_dir", "temp_images")) / file_md5
    if not cache_dir.is_dir():
        return None

    img_files = sorted(
        cache_dir.glob("page_*.jp*g"),
        key=lambda p: int(re.search(r"page_(\d+)", p.stem).group(1)) if re.search(r"page_(\d+)", p.stem) else p.name,
    )
    if not img_files:
        return None

    pages: list[dict[str, str]] = []
    for f in img_files:
        b64 = base64.b64encode(f.read_bytes()).decode("ascii")
        pages.append({"content_b64": b64, "mime_type": "image/jpeg"})

    logger.info("[file_utils] [Cache Hit] 命中 MD5 快取 (%s)，直接從暫存資料夾載入 %d 頁影像！", file_md5, len(pages))
    return pages


def _save_cached_images(
    file_md5: str,
    pages: list[dict[str, str]],
    raw_pdf_path: Path | None,
    filename: str,
    settings: Settings,
) -> None:
    """若啟用 vlm_save_temp_images，將轉檔後的 JPEG 圖片與中介 PDF 保存至 temp_images/{file_md5}/。"""
    if not getattr(settings, "vlm_save_temp_images", False) or not pages:
        return
    try:
        cache_dir = Path(getattr(settings, "vlm_temp_images_dir", "temp_images")) / file_md5
        cache_dir.mkdir(parents=True, exist_ok=True)

        for idx, page in enumerate(pages, start=1):
            img_path = cache_dir / f"page_{idx:03d}.jpg"
            img_bytes = base64.b64decode(page["content_b64"])
            img_path.write_bytes(img_bytes)

        if raw_pdf_path and raw_pdf_path.exists():
            target_pdf = cache_dir / "converted.pdf"
            shutil.copy2(raw_pdf_path, target_pdf)

        meta_path = cache_dir / "meta.json"
        meta_info = {
            "file_md5": file_md5,
            "filename": filename,
            "total_pages": len(pages),
            "created_at": datetime.datetime.now().isoformat(),
        }
        meta_path.write_text(json.dumps(meta_info, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[file_utils] [Cache Save] 已將 %d 頁轉檔影像儲存至 MD5 快取資料夾：%s", len(pages), cache_dir)
    except Exception as e:
        logger.warning("[file_utils] [Cache Save 異常] 儲存快取影像失敗：%s", e)


async def expand_to_image_pages(
    raw: bytes,
    mime_type: str,
    filename: str,
    *,
    worker: LibreOfficeWorker | None = None,
) -> list[dict[str, str]]:
    """將上傳附件展開為可餵入 VLM 的圖片頁面清單。

    分派規則（依副檔名優先，MIME 類型為補充判斷）：

    - PDF（.pdf）        → 逐頁轉 JPEG base64
    - Office 文件        → LibreOffice 轉 PDF，再逐頁轉 JPEG base64
      (.doc/.docx/.odt/.rtf / .xls/.xlsx/.ods / .ppt/.pptx/.odp)
      有 worker 時走 UNO 長駐池，無 worker 時 fallback 冷啟動
    - 圖片（.jpg/.png 等）→ 修正 MIME 類型後直接回傳單頁
    - 其他               → 原樣包成單頁回傳（MIME 不修正）

    Args:
        raw:      上傳檔案原始位元組。
        mime_type: 上游提供的 MIME 類型字串（可能不準確，副檔名優先）。
        filename:  原始檔名（用於副檔名辨識）。
        worker:   LibreOfficeWorker 實例；提供時走 UNO 池模式，None 時退回冷啟動。

    Returns:
        list of dicts，每個 dict 包含 ``content_b64`` 與 ``mime_type``。
    """
    settings = get_settings()
    file_md5 = hashlib.md5(raw).hexdigest()

    # 優先嘗試 MD5 快取命中
    cached = _try_load_cached_images(file_md5, settings)
    if cached is not None:
        return cached

    loop = asyncio.get_event_loop()
    ext = Path(filename).suffix.lower()

    # ── PDF ──────────────────────────────────────────────────────────────────
    if ext == ".pdf" or mime_type == _PDF_MIME:
        logger.info("[file_utils] [PDF轉檔] 開始逐頁渲染為圖片：檔名=%s  大小=%d bytes", filename, len(raw))
        t0 = asyncio.get_event_loop().time()
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "input.pdf"
            pdf_path.write_bytes(raw)
            pages = await loop.run_in_executor(None, _pdf_to_image_pages, pdf_path)
            _save_cached_images(file_md5, pages, pdf_path, filename, settings)
        elapsed = (asyncio.get_event_loop().time() - t0) * 1000
        logger.info("[file_utils] [PDF轉檔完成] 檔名=%s  總頁數=%d 頁  耗時=%.1f ms", filename, len(pages), elapsed)
        return pages

    # ── Office 文件 → PDF → JPEG ─────────────────────────────────────────────
    if ext in _OFFICE_EXTS:
        logger.info("[file_utils] [Office轉檔] 格式=%s  檔名=%s  開始轉換為 PDF 及頁面影像", ext, filename)
        t0 = asyncio.get_event_loop().time()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # 以原始副檔名保存，確保 LibreOffice 正確識別格式
            src_path = tmp_path / f"input{ext}"
            src_path.write_bytes(raw)
            logger.debug("[file_utils] 已保存上傳文件至暫存：%s", src_path)
            
            t_lo_start = asyncio.get_event_loop().time()
            if worker is not None:
                logger.info("[file_utils] [LibreOffice] 使用 Daemon Worker 池進行轉檔...")
                pdf_path = await worker.convert(src_path, tmp_path)
            else:
                logger.info("[file_utils] [LibreOffice] 啟動 LibreOffice Portable (headless) 進行轉檔...")
                pdf_path = await loop.run_in_executor(
                    None, _docx_to_pdf_via_libreoffice, src_path, tmp_path
                )
            t_lo_elapsed = (asyncio.get_event_loop().time() - t_lo_start) * 1000
            logger.info("[file_utils] [LibreOffice完成] PDF 產出路徑=%s  LibreOffice耗時=%.1f ms", pdf_path.name, t_lo_elapsed)

            t_img_start = asyncio.get_event_loop().time()

            vlm_dpi = settings.vlm_dpi or 150
            logger.info("[file_utils] [Poppler] 開始將 PDF 逐頁渲染為 JPEG 影像 (DPI=%d)...", vlm_dpi)
            pages = await loop.run_in_executor(None, _pdf_to_image_pages, pdf_path, vlm_dpi)
            t_img_elapsed = (asyncio.get_event_loop().time() - t_img_start) * 1000
            logger.info("[file_utils] [Poppler完成] 總頁數=%d 頁  影像渲染耗時=%.1f ms", len(pages), t_img_elapsed)

            _save_cached_images(file_md5, pages, pdf_path, filename, settings)

        total_elapsed = (asyncio.get_event_loop().time() - t0) * 1000
        logger.info("[file_utils] [Office全流程完成] 檔名=%s  總頁數=%d 頁  總耗時=%.1f ms", filename, len(pages), total_elapsed)
        return pages

    # ── 圖片格式 → 修正 MIME 直接回傳 ───────────────────────────────────────
    if ext in _IMAGE_EXT_TO_MIME:
        corrected_mime = _IMAGE_EXT_TO_MIME[ext]
        logger.debug("[file_utils] 圖片附件（%s mime=%s）：%s", ext, corrected_mime, filename)
        pages = [{"content_b64": base64.b64encode(raw).decode("ascii"), "mime_type": corrected_mime}]
        _save_cached_images(file_md5, pages, None, filename, settings)
        return pages

    # ── 其他格式：原樣回傳 ───────────────────────────────────────────────────
    logger.warning(
        "[file_utils] 未知格式（%s），原樣傳入 VLM（mime=%s）：%s", ext, mime_type, filename
    )
    return [{"content_b64": base64.b64encode(raw).decode("ascii"), "mime_type": mime_type}]
