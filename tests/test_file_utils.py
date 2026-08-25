import base64
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.config import Settings
from app.lo.file_utils import (
    _save_cached_images,
    _try_load_cached_images,
    expand_to_image_pages,
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


if __name__ == "__main__":
    unittest.main()
