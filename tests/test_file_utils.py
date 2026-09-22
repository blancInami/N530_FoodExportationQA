import asyncio
import base64
import contextlib
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.config import Settings
from app.lo import file_utils
from app.lo.file_utils import (
    _save_cached_images,
    _try_load_cached_images,
    expand_to_image_pages,
    resolve_poppler_bin_dir,
)


class FileUtilsCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.settings = Settings(
            vlm_save_temp_images=True,
            vlm_temp_images_dir=self.tmp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load_cached_images(self):
        file_bytes = b"sample document content for md5 cache test"
        file_md5 = hashlib.md5(file_bytes).hexdigest()

        # 模擬 2 頁圖片
        img1_b64 = base64.b64encode(b"fake_jpeg_page_1").decode("ascii")
        img2_b64 = base64.b64encode(b"fake_jpeg_page_2").decode("ascii")
        pages = [
            {"content_b64": img1_b64, "mime_type": "image/jpeg"},
            {"content_b64": img2_b64, "mime_type": "image/jpeg"},
        ]

        # 1. 儲存快取
        _save_cached_images(file_md5, pages, None, "test.docx", self.settings)

        cache_dir = Path(self.tmp_dir) / file_md5
        self.assertTrue(cache_dir.is_dir())
        self.assertTrue((cache_dir / "page_001.jpg").exists())
        self.assertTrue((cache_dir / "page_002.jpg").exists())
        self.assertTrue((cache_dir / "meta.json").exists())

        # 2. 讀取快取（Cache Hit）
        loaded_pages = _try_load_cached_images(file_md5, self.settings)
        self.assertIsNotNone(loaded_pages)
        self.assertEqual(len(loaded_pages), 2)
        self.assertEqual(loaded_pages[0]["content_b64"], img1_b64)
        self.assertEqual(loaded_pages[1]["content_b64"], img2_b64)

    def test_cache_miss_when_disabled(self):
        disabled_settings = Settings(
            vlm_save_temp_images=False,
            vlm_temp_images_dir=self.tmp_dir,
        )
        file_md5 = "non_existent_md5"
        self.assertIsNone(_try_load_cached_images(file_md5, disabled_settings))

    @patch("app.lo.file_utils.get_settings")
    async def test_expand_to_image_pages_cache_hit_bypasses_conversion(self, mock_settings):
        mock_settings.return_value = self.settings

        raw_data = b"cache_hit_test_payload"
        file_md5 = hashlib.md5(raw_data).hexdigest()

        # 預先在暫存資料夾寫入快取
        cache_dir = Path(self.tmp_dir) / file_md5
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "page_001.jpg").write_bytes(b"cached_image_data")

        # 呼叫 expand_to_image_pages
        pages = await expand_to_image_pages(raw_data, "application/pdf", "test.pdf")

        self.assertEqual(len(pages), 1)
        expected_b64 = base64.b64encode(b"cached_image_data").decode("ascii")
        self.assertEqual(pages[0]["content_b64"], expected_b64)


class _FakeWorker:
    worker_id = 0

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    async def convert(self, src: Path, out_dir: Path) -> Path:
        self.calls += 1
        if self.fail:
            raise RuntimeError("UNO 轉換失敗")
        pdf = out_dir / (src.stem + ".pdf")
        pdf.write_bytes(b"%PDF-fake")
        return pdf


class _FakePool:
    def __init__(self, worker: _FakeWorker | None = None, timeout: bool = False):
        self.worker = worker
        self.timeout = timeout

    @contextlib.asynccontextmanager
    async def acquire_context(self, timeout=None):
        if self.timeout:
            raise asyncio.TimeoutError
        yield self.worker


def _fake_cold_convert(src: Path, out_dir: Path) -> Path:
    pdf = out_dir / (src.stem + ".pdf")
    pdf.write_bytes(b"%PDF-cold")
    return pdf


class FileUtilsLibreOfficePoolTests(unittest.IsolatedAsyncioTestCase):
    """Office 轉檔路徑：全域資源池借用、逾時與失敗時退回冷啟動。"""

    def setUp(self):
        self.settings = Settings(vlm_save_temp_images=False, lo_acquire_timeout=1.0)
        self.pages = [{"content_b64": "eA==", "mime_type": "image/jpeg"}]

    async def _expand(self, pool):
        with patch("app.lo.file_utils.get_settings", return_value=self.settings),              patch("app.lo.file_utils.get_lo_pool", return_value=pool),              patch("app.lo.file_utils._pdf_to_image_pages", return_value=self.pages),              patch("app.lo.file_utils._docx_to_pdf_via_libreoffice", side_effect=_fake_cold_convert) as cold:
            pages = await expand_to_image_pages(b"docx-bytes", "", "sample.docx")
        return pages, cold

    async def test_uses_pool_worker_when_enabled(self):
        worker = _FakeWorker()
        pages, cold = await self._expand(_FakePool(worker))
        self.assertEqual(pages, self.pages)
        self.assertEqual(worker.calls, 1)
        cold.assert_not_called()

    async def test_cold_start_when_pool_disabled(self):
        pages, cold = await self._expand(None)
        self.assertEqual(pages, self.pages)
        cold.assert_called_once()

    async def test_falls_back_to_cold_start_on_acquire_timeout(self):
        pages, cold = await self._expand(_FakePool(timeout=True))
        self.assertEqual(pages, self.pages)
        cold.assert_called_once()

    async def test_falls_back_to_cold_start_on_worker_failure(self):
        worker = _FakeWorker(fail=True)
        pages, cold = await self._expand(_FakePool(worker))
        self.assertEqual(pages, self.pages)
        self.assertEqual(worker.calls, 1)
        cold.assert_called_once()



class ResolvePopplerBinDirTests(unittest.TestCase):
    """POPPLER_DIR 解析：預設 tools 目錄、Library/bin / bin / 直接指定 bin 目錄結構。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_bin(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        (path / "pdftoppm.exe").write_bytes(b"")
        return path

    def test_default_uses_tools_poppler(self):
        expected = file_utils._DEFAULT_POPPLER_DIR / "bin"
        self.assertEqual(resolve_poppler_bin_dir(""), expected)

    def test_release_zip_layout(self):
        bin_dir = self._make_bin(self.tmp / "poppler-24.08.0" / "Library" / "bin")
        self.assertEqual(resolve_poppler_bin_dir(str(self.tmp / "poppler-24.08.0")), bin_dir)

    def test_bin_layout(self):
        bin_dir = self._make_bin(self.tmp / "poppler" / "bin")
        self.assertEqual(resolve_poppler_bin_dir(str(self.tmp / "poppler")), bin_dir)

    def test_bin_dir_directly(self):
        bin_dir = self._make_bin(self.tmp / "bin")
        self.assertEqual(resolve_poppler_bin_dir(str(bin_dir)), bin_dir)

    def test_relative_path_is_based_on_project_root(self):
        self.assertEqual(resolve_poppler_bin_dir("some/poppler"), file_utils._PROJECT_ROOT / "some" / "poppler" / "bin")

    def test_pdf_to_image_pages_raises_when_poppler_missing(self):
        settings = Settings(poppler_dir=str(self.tmp / "missing"))
        with patch("app.config.get_settings", return_value=settings):
            with self.assertRaisesRegex(FileNotFoundError, "POPPLER_DIR"):
                file_utils._pdf_to_image_pages(self.tmp / "x.pdf")


if __name__ == "__main__":
    unittest.main()
