"""
模組四：全域詞彙聚合統計與 CSV 匯出

功能：
    將所有文件對中通過反向驗證的詞彙對進行全域統計，
    計算每一組 (中文字詞, 英文字詞) 的出現頻率，
    並匯出為按頻率降冪排序的 CSV 檔案。

CSV 格式：
    欄位：中文字詞、英文字詞、出現頻率
    編碼：utf-8-sig（含 BOM，確保 Windows Excel 正確顯示中文）
"""

import csv
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


# ── 聚合函式 ──────────────────────────────────────────────────────────────────

def aggregate_terms(all_terms: list[dict]) -> list[dict]:
    """
    對所有已通過驗證的詞彙對進行全域頻率統計。

    統計 key 為 (chinese_term, english_term) tuple，
    相同的雙語對應關係計為同一筆，累計出現次數，
    並收集所有相關的負責單位（拆分後取唯一值）。

    Args:
        all_terms: 所有文件的通過驗證詞彙列表
                   每個 dict 含 'chinese_term', 'english_term' 與 'responsible_unit'

    Returns:
        list[dict]，按出現頻率「降冪」排序，每個 dict 格式：
        {
            "中文字詞": str,
            "英文字詞": str,
            "出現頻率": int,
            "負責單位": list[str],
        }
    """
    # if not all_terms:
    #     logger.warning("無任何通過驗證的詞彙，回傳空列表")
    #     return []

    # 負責單位拆分用的正規表達式（支援 、 , / ; 等常用分隔符）
    unit_splitter = re.compile(r"[、,/;/\s]+")

    # 以 (中文詞, 英文詞) 為 key 統計出現次數與收集單位
    # stats 結構: {(zh, en): {"count": int, "units": set}}
    stats: dict[tuple[str, str], dict] = {}
    for item in all_terms:
        zh = item.get("chinese_term", "").strip()
        en = item.get("english_term", "").strip()
        unit_str = str(item.get("responsible_unit", "")).strip()

        if zh and en:
            key = (zh, en.lower())
            if key not in stats:
                stats[key] = {"count": 0, "units": set(), "sources": set()}
            stats[key]["count"] += 1

            source_str = str(item.get("source_file", "")).strip()
            if source_str:
                stats[key]["sources"].add(source_str)

            if unit_str:
                # 拆分可能包含多個單位的字串並去重存入 set
                parts = [p.strip() for p in unit_splitter.split(unit_str) if p.strip()]
                stats[key]["units"].update(parts)

    # 依頻率降冪排序
    sorted_pairs = sorted(
        stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )

    result = [
        {
            "中文字詞": zh,
            "英文字詞": en,
            "出現頻率": data["count"],
            "負責單位": sorted(list(data["units"])),
            "來源檔案": sorted(list(data["sources"])),
        }
        for (zh, en), data in sorted_pairs
    ]

    logger.info(
        "詞彙聚合完成：輸入 %d 筆原始詞彙  去重後共 %d 組獨特詞彙對",
        len(all_terms), len(result),
    )
    return result


# ── CSV 匯出函式 ──────────────────────────────────────────────────────────────

def export_csv(terms: list[dict], output_path: str) -> None:
    """
    將聚合後的詞彙統計結果匯出為 CSV 檔案。

    CSV 規格：
    - 欄位順序：中文字詞、英文字詞、出現頻率、負責單位
    - 編碼：utf-8-sig（含 BOM），確保 Microsoft Excel 正確識別 UTF-8 中文
    - 含表頭（header row）
    - 已依頻率降冪排序（呼叫前由 aggregate_terms 處理）

    Args:
        terms      : 由 aggregate_terms() 回傳的結果列表
        output_path: CSV 輸出路徑（字串，如 "terms_output.csv"）

    Raises:
        此函式不拋出例外，失敗時記錄 error 並靜默返回。
    """
    if not terms:
        logger.warning("詞彙列表為空，不建立 CSV 檔案：%s", output_path)
        return

    output = Path(output_path)

    # 自動建立輸出目錄（若不存在）
    output.parent.mkdir(parents=True, exist_ok=True)

    # 將 '負責單位' list 轉為逗號分隔字串以便輸出至 CSV
    csv_rows = []
    for t in terms:
        row = t.copy()
        if isinstance(row.get("負責單位"), list):
            row["負責單位"] = "、".join(row["負責單位"])
        if isinstance(row.get("來源檔案"), list):
            row["來源檔案"] = ", ".join(row["來源檔案"])
        csv_rows.append(row)

    fieldnames = ["中文字詞", "英文字詞", "出現頻率", "負責單位", "來源檔案"]

    try:
        # newline="" 防止 Windows 上 csv.writer 產生多餘空行
        # encoding="utf-8-sig" 加入 BOM，Excel 開啟時不亂碼
        with open(output, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.info(
            "CSV 已匯出：%s  共 %d 筆詞彙對",
            output.resolve(), len(terms),
        )

    except OSError as exc:
        logger.error("CSV 匯出失敗：%s  錯誤：%s", output_path, exc)
