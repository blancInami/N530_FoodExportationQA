"""
Utility functions for text processing.
"""
import re


# 預編譯：匹配全半形空白、中英文標點、零寬字元
_NORMALIZE_PATTERN = re.compile(
    r"[\s\u3000"                       # 全半形空白
    r"\uff0c\u3002\u3001\uff1b\uff1a\uff01\uff1f"  # ，。、；：！？
    r"\uff08\uff09\u300c\u300d\u300e\u300f\u3010\u3011"  # （）「」『』【】
    r"\u2018\u2019\u201c\u201d\u300a\u300b\u3008\u3009"  # ''""《》〈〉
    r".,;:!?()\[\]{}\\/\-_@#$%^&*+=|<>~`'\""  # ASCII 標點
    r"\u200b-\u200f\ufeff"             # 零寬字元 + BOM
    r"]+"
)


def normalize_for_matching(text: str) -> str:
    """
    正規化文字以供 Aho-Corasick 比對使用。
    剔除全半形空白、常見中英標點、零寬字元，回傳連續純文字字串。

    目的：讓口語輸入（如「牛 肉」「牛、肉」）與自動機模式（「牛肉」）能夠匹配。
    """
    if not text:
        return ""
    return _NORMALIZE_PATTERN.sub("", text)


def sanitize_text_for_db(text: str) -> str:
    """
    Remove control characters (except newline/tab) that may cause
    issues when writing to PostgreSQL.
    """
    if not text:
        return ""
    # Remove ASCII control characters except \n (0x0A) and \t (0x09)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Remove null bytes explicitly
    cleaned = cleaned.replace("\x00", "")
    return cleaned.strip()


def sliding_window_chunk(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> list[str]:
    """
    Sliding window chunking with overlap.
    Splits text into chunks of `chunk_size` characters with `overlap` overlap.
    """
    if not text:
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap

    return chunks
