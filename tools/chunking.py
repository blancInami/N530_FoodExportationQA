"""
模組二（Part B）：文本分塊

功能：
    將長文本切分為適合 LLM 推論視窗的段落，採用滑動窗口機制，
    並提供中英文段落的比例索引對齊方法。

設計說明：
    - chunk_size 預設 2000 字元（較大，適合名詞擷取場景，避免切斷上下文）
    - 驗證時使用全文比對（非 chunk），因此此分塊僅用於 LLM 推論的輸入視窗
    - 對齊邏輯採「索引比例映射」，不強求中英文 chunk 數量相等
"""

import logging

logger = logging.getLogger(__name__)


# ── 分塊函式 ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[str]:
    """
    使用滑動窗口機制將文本切分為多個段落。

    切分邏輯：
    - 若文本長度 <= chunk_size，直接回傳含整段文本的單元素列表
    - 每次前進 (chunk_size - overlap) 個字元，保留末尾 overlap 字元作為上下文銜接
    - 去除空白段落

    Args:
        text      : 要切分的文字內容（已萃取的 Markdown 純文字）
        chunk_size: 每個切塊的最大字元數（預設 2000）
        overlap   : 相鄰切塊的重疊字元數（預設 200）

    Returns:
        list[str]，每個元素為一個文本段落。
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # 文本長度未超過上限，直接作為單一 chunk
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap  # 每次前進的步長

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    logger.debug(
        "文本分塊完成：原始長度=%d  chunk_size=%d  overlap=%d  共 %d 塊",
        len(text), chunk_size, overlap, len(chunks),
    )
    return chunks


# ── 對齊函式 ──────────────────────────────────────────────────────────────────

def align_chunks(
    zh_chunks: list[str],
    en_chunks: list[str],
) -> list[tuple[str, str]]:
    """
    將中英文 chunk 列表以索引比例對齊，配對成 (中文段落, 英文段落) tuple 列表。

    對齊邏輯（比例索引映射）：
        以較多 chunk 數的一方為基準（共 N 對）。
        對第 i 對（0-indexed），取：
            zh_chunks[i * len(zh_chunks) // N]
            en_chunks[i * len(en_chunks) // N]
        藉此確保每一對都有對應內容，不會因數量不同而遺漏。

    使用場景說明：
        中英文文本的段落數量可能因翻譯風格差異而不同，
        此方法透過比例映射確保兩側的語意覆蓋範圍盡量對應，
        是一種「近似語意對齊」的簡單實作。

    【特殊情況】
        若某一側文本完全為空（chunks 為空列表），
        回傳空列表並記錄 warning。

    Args:
        zh_chunks: 中文文本的 chunk 列表
        en_chunks: 英文文本的 chunk 列表

    Returns:
        list[tuple[str, str]]，每個 tuple 為 (中文段落, 英文段落)
    """
    if not zh_chunks:
        logger.warning("中文 chunks 為空，無法對齊")
        return []
    if not en_chunks:
        logger.warning("英文 chunks 為空，無法對齊")
        return []

    zh_n = len(zh_chunks)
    en_n = len(en_chunks)
    # 以較大的 chunk 數量為總配對數，確保覆蓋較多的文本
    total = max(zh_n, en_n)

    pairs: list[tuple[str, str]] = []
    for i in range(total):
        # 比例映射索引：確保不超出各自列表邊界
        zh_idx = i * zh_n // total
        en_idx = i * en_n // total
        pairs.append((zh_chunks[zh_idx], en_chunks[en_idx]))

    logger.debug(
        "Chunk 對齊完成：中文 %d 塊  英文 %d 塊  → 共 %d 對",
        zh_n, en_n, len(pairs),
    )
    return pairs
