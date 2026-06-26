"""
Markdown conversion wrapper for tools/ingest_agent.py.

Converts PDF, DOCX, DOC, XLSX, XLS files to Markdown via markitdown.
All functions are synchronous (intended for threading in async contexts).
"""
from pathlib import Path


def convert_to_markdown(file_path: str) -> str:
    """
    Convert a document file to Markdown string.

    Supports: .pdf, .docx, .doc, .xlsx, .xls

    Returns:
        str: Markdown-formatted text content.

    Raises:
        ValueError: If the file format is unsupported or conversion fails.
    """
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")

    supported = {"pdf", "docx", "doc", "xlsx", "xls"}
    if ext not in supported:
        raise ValueError(f"不支援的檔案格式 .{ext}。允許格式：{', '.join(sorted(supported))}")

    try:
        from markitdown import MarkItDown  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "markitdown 套件未安裝。請執行：pip install markitdown"
        ) from exc

    md = MarkItDown()
    result = md.convert(str(path))
    text = result.text_content or ""
    if not text.strip():
        raise ValueError(f"檔案轉換後內容為空：{file_path}")
    return text
