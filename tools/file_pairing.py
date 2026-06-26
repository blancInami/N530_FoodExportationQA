"""
模組一（Part A）：資料夾走訪與中英文 PDF 配對

功能：
    走訪指定資料夾，根據檔名後綴（_EN / _CH）自動將
    同一份文件的中英文版本配對為一組 (Path_中文, Path_英文)。

配對規則（大小寫不敏感）：
    假設檔名格式為：<前綴>_EN.pdf 與 <前綴>_CH.pdf
    其中後綴標識支援：
        中文版：_CH、_ZH、_CHS、_CHT、_中文
        英文版：_EN、_ENG、_英文

範例：
    doc_EN.pdf + doc_CH.pdf     → (doc_CH.pdf, doc_EN.pdf)
    report_ENG.pdf + report_ZH.pdf → (report_ZH.pdf, report_ENG.pdf)
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 後綴分類常數 ──────────────────────────────────────────────────────────────

# 代表中文版本的後綴標識（統一轉小寫後比對）
_CHINESE_SUFFIXES: set[str] = {"_ch", "_zh", "_chs", "_cht", "_中文", "_繁中", "_簡中"}

# 代表英文版本的後綴標識（統一轉小寫後比對）
_ENGLISH_SUFFIXES: set[str] = {"_en", "_eng", "_英文"}


# ── 主要函式 ──────────────────────────────────────────────────────────────────

def discover_pdf_pairs(folder_path: str) -> list[tuple[Path, Path]]:
    """
    走訪指定資料夾，找出所有可配對的中英文 PDF 組合。

    Args:
        folder_path: 要掃描的資料夾路徑字串（可為相對或絕對路徑）

    Returns:
        list[tuple[Path, Path]]
        每個 tuple 格式為 (中文PDF路徑, 英文PDF路徑)。
        無法配對的 PDF 會記錄 warning 並排除於結果之外。

    Raises:
        FileNotFoundError: 若指定資料夾不存在
        NotADirectoryError: 若路徑不是資料夾
    """
    folder = Path(folder_path)

    # 驗證資料夾存在性
    if not folder.exists():
        raise FileNotFoundError(f"指定的資料夾不存在：{folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"指定路徑不是資料夾：{folder}")

    # 遞迴（rglob）或僅掃描頂層（glob）。
    # 此處採頂層掃描（glob），若需遞迴請改為 rglob("*.pdf")。
    all_pdfs = list(folder.glob("*.pdf"))
    logger.info("資料夾 %s 共找到 %d 個 PDF 檔案", folder, len(all_pdfs))

    if not all_pdfs:
        logger.warning("資料夾頂層沒有任何 PDF 檔案，嘗試遞迴走訪子資料夾：%s", folder)
        subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
        if not subdirs:
            logger.warning("資料夾中沒有任何子資料夾：%s", folder)
            return []
        all_pairs: list[tuple[Path, Path]] = []
        for subdir in subdirs:
            try:
                sub_pairs = discover_pdf_pairs(str(subdir))
                all_pairs.extend(sub_pairs)
            except (FileNotFoundError, NotADirectoryError) as exc:
                logger.warning("子資料夾處理失敗，跳過：%s  錯誤：%s", subdir, exc)
        return all_pairs

    # 將所有 PDF 分類：key = 共同前綴，value = {"zh": Path, "en": Path}
    groups: dict[str, dict[str, Path]] = {}

    for pdf_path in all_pdfs:
        stem = pdf_path.stem          # 去掉 .pdf 的檔名，如 "doc_EN"
        prefix, lang = _split_stem(stem)

        if lang is None:
            # 無法識別語言標識，跳過此檔案
            logger.warning("無法識別語言後綴，跳過：%s", pdf_path.name)
            continue

        if prefix not in groups:
            groups[prefix] = {}

        if lang in groups[prefix]:
            # 同一前綴下同語言出現兩次，取後者並記錄警告
            logger.warning(
                "前綴 '%s' 的 %s 版本有重複，後者覆蓋前者：新=%s  舊=%s",
                prefix, lang, pdf_path.name, groups[prefix][lang].name,
            )

        groups[prefix][lang] = pdf_path

    # 組合配對結果
    pairs: list[tuple[Path, Path]] = []

    for prefix, lang_map in groups.items():
        zh_path = lang_map.get("zh")
        en_path = lang_map.get("en")

        if zh_path and en_path:
            pairs.append((zh_path, en_path))
            logger.info("配對成功：中文=%s  英文=%s", zh_path.name, en_path.name)
        else:
            # 只有單一語言版本，無法配對
            existing = zh_path or en_path
            missing_lang = "英文" if zh_path else "中文"
            logger.warning(
                "前綴 '%s' 缺少 %s 版本，無法配對，跳過：%s",
                prefix, missing_lang, existing.name if existing else "(未知)",
            )

    logger.info("共成功配對 %d 組中英文 PDF", len(pairs))
    return pairs


# ── 輔助函式 ──────────────────────────────────────────────────────────────────

def _split_stem(stem: str) -> tuple[str, str | None]:
    """
    從檔名主體（不含副檔名）中分離出「共同前綴」與「語言標識」。

    匹配策略：
    以已知的語言後綴列表比對 stem 尾端，找到最長匹配的後綴。
    例如：
        "doc_EN"    → ("doc", "en")
        "report_ZH" → ("report", "zh")
        "unknown"   → ("unknown", None)

    Args:
        stem: 檔名去掉副檔名的部分，例如 "doc_EN"

    Returns:
        (prefix, lang) tuple，其中：
        - prefix: 去掉語言後綴的共同前綴
        - lang  : "zh"（中文）或 "en"（英文）或 None（無法識別）
    """
    stem_lower = stem.lower()

    # 按後綴長度由長到短嘗試，確保長後綴優先匹配（如 _cht 優先於 _ch）
    all_suffixes = sorted(
        [(_s, "zh") for _s in _CHINESE_SUFFIXES] +
        [(_s, "en") for _s in _ENGLISH_SUFFIXES],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    for suffix, lang in all_suffixes:
        if stem_lower.endswith(suffix):
            # 取得共同前綴（保留原始大小寫）
            prefix = stem[: len(stem) - len(suffix)]
            return prefix, lang

    # 也嘗試用正則匹配括號標識，例如 "doc (英文)" 或 "doc(EN)"
    bracket_match = re.search(r"[（(](en|eng|英文|ch|zh|中文|繁中|chs|cht)[）)]$", stem_lower)
    if bracket_match:
        matched_tag = bracket_match.group(1)
        prefix = stem[: bracket_match.start()]
        if matched_tag in {s.lstrip("_") for s in _CHINESE_SUFFIXES}:
            return prefix.strip(), "zh"
        return prefix.strip(), "en"

    return stem, None  # 無法識別語言標識
