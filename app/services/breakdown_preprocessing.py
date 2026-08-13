"""Canonical text preparation for the breakdown pipeline."""
import html
import re
import unicodedata


_FENCED_CODE_RE = re.compile(r"^\s*```")
_MARKDOWN_LINK_TARGET_RE = re.compile(r"\]\([^)]+\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ALIGNMENT_CELL_RE = re.compile(r"^:?-{3,}:?$")
_QUESTION_ID_RE = re.compile(r"^\d+(?:\.\d+)*$")


def normalize_markdown_for_breakdown(
    markdown_text: str,
    *,
    linearize_tables: bool = True,
) -> str:
    """Normalize visible text and optionally linearize Markdown tables.

    URLs, HTML/XML tags, and fenced code blocks remain byte-for-byte unchanged.
    """
    normalized_lines: list[str] = []
    in_fenced_code = False
    for line in markdown_text.splitlines(keepends=True):
        if _FENCED_CODE_RE.match(line):
            in_fenced_code = not in_fenced_code
            normalized_lines.append(line)
        elif in_fenced_code:
            normalized_lines.append(line)
        else:
            normalized_lines.append(_normalize_visible_text(line))

    normalized = "".join(normalized_lines)
    return _linearize_markdown_tables(normalized) if linearize_tables else normalized


def _normalize_visible_text(text: str) -> str:
    """Apply NFKC outside syntax that must retain its original bytes."""
    protected = re.compile(
        f"({_MARKDOWN_LINK_TARGET_RE.pattern}|{_HTML_TAG_RE.pattern})"
    )
    parts = protected.split(text)
    return "".join(
        part if index % 2 else unicodedata.normalize("NFKC", part)
        for index, part in enumerate(parts)
    )


def _linearize_markdown_tables(markdown_text: str) -> str:
    """Convert contiguous Markdown tables to line-oriented XML-like rows."""
    output: list[str] = []
    lines = markdown_text.splitlines(keepends=True)
    index = 0

    while index < len(lines):
        cells = _split_table_row(lines[index])
        if not cells:
            output.append(lines[index])
            index += 1
            continue

        table_rows: list[list[str]] = []
        while index < len(lines):
            row = _split_table_row(lines[index])
            if not row:
                break
            if not _is_alignment_row(row):
                table_rows.append(row)
            index += 1

        output.extend(_linearize_rows(table_rows))

    return "".join(output)


def _split_table_row(line: str) -> list[str]:
    """Split one pipe-delimited Markdown row while respecting escaped pipes."""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []

    cells: list[str] = []
    buffer: list[str] = []
    escaped = False
    for character in stripped[1:-1]:
        if escaped:
            buffer.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(character)
    if escaped:
        buffer.append("\\")
    cells.append("".join(buffer).strip())
    return cells


def _is_alignment_row(cells: list[str]) -> bool:
    return bool(cells) and all(_ALIGNMENT_CELL_RE.fullmatch(cell) for cell in cells)


def _linearize_rows(rows: list[list[str]]) -> list[str]:
    """Emit stable row and cell tags without guessing unreliable column meanings."""
    if not rows:
        return []

    output: list[str] = []
    for row_index, cells in enumerate(rows, start=1):
        roles = _infer_cell_roles(cells)
        cell_xml = "".join(
            f'<Cell column="{column_index}" role="{role}">{html.escape(cell)}</Cell>'
            for column_index, (cell, role) in enumerate(zip(cells, roles), start=1)
        )
        output.append(f'<Row index="{row_index}">{cell_xml}</Row>\n')
    return output


def _infer_cell_roles(cells: list[str]) -> list[str]:
    """Use semantic roles only for the unambiguous ID/source/target row shape."""
    if (
        len(cells) == 3
        and _QUESTION_ID_RE.fullmatch(cells[0])
        and _contains_non_latin(cells[1])
        and _contains_latin(cells[2])
    ):
        return ["id", "source", "target"]
    return ["unknown"] * len(cells)


def _contains_latin(text: str) -> bool:
    return any(character.isascii() and character.isalpha() for character in text)


def _contains_non_latin(text: str) -> bool:
    return any(not character.isascii() and character.isalpha() for character in text)