import tempfile
import unittest
from pathlib import Path

from app.lo import lo_pool
from app.lo.lo_pool import resolve_lo_program_dir


def _make_program_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "soffice.exe").write_bytes(b"")
    return path


class ResolveLoProgramDirTests(unittest.TestCase):
    """LIBREOFFICE_DIR 解析：預設 tools 目錄、Portable / 安裝版 / program 目錄結構。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_default_uses_tools_portable(self):
        expected = lo_pool._DEFAULT_LO_DIR / "App" / "libreoffice" / "program"
        self.assertEqual(resolve_lo_program_dir(""), expected)
        self.assertEqual(resolve_lo_program_dir("   "), expected)

    def test_portable_layout(self):
        program = _make_program_dir(self.tmp / "LOPortable" / "App" / "libreoffice" / "program")
        self.assertEqual(resolve_lo_program_dir(str(self.tmp / "LOPortable")), program)

    def test_installed_layout(self):
        program = _make_program_dir(self.tmp / "LibreOffice" / "program")
        self.assertEqual(resolve_lo_program_dir(str(self.tmp / "LibreOffice")), program)

    def test_program_dir_directly(self):
        program = _make_program_dir(self.tmp / "program")
        self.assertEqual(resolve_lo_program_dir(str(program)), program)

    def test_relative_path_is_based_on_project_root(self):
        self.assertEqual(
            resolve_lo_program_dir("some/lo"),
            lo_pool._PROJECT_ROOT / "some" / "lo" / "App" / "libreoffice" / "program",
        )

    def test_missing_dir_falls_back_to_portable_layout_path(self):
        missing = self.tmp / "missing"
        self.assertEqual(resolve_lo_program_dir(str(missing)), missing / "App" / "libreoffice" / "program")


if __name__ == "__main__":
    unittest.main()
