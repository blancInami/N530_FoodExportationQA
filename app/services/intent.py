"""
Intent classification service — 雙層分類架構。

Layer 1（快速路徑）：Aho-Corasick 關鍵字比對
  - 從「單位對照表」分工關鍵字欄位建立自動機
  - 搭配 alias_mapping.json 別名擴展（由 tools/extract_aliases.py 離線產出）
  - O(n) 時間複雜度，無 LLM 呼叫，無額外延遲

Layer 2（LLM 兜底）：llm_fallback_classify()
  - 僅在 Aho-Corasick 無命中時觸發
  - 將「單位對照表」全量列（行數少，< 50）完整提供給 LLM
  - LLM 從清單中選出最合適的負責機關與單位
  - 若 LLM 判斷無關聯 → 回傳 None
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

try:
    import ahocorasick_rs as ahocorasick
    _USE_AHO = True
except ImportError:
    _USE_AHO = False

from app.models import 單位對照表
from app.services.llm import chat_completion
from app.utils import normalize_for_matching

logger = logging.getLogger(__name__)

# 別名映射 JSON 的預設路徑（由 tools/extract_aliases.py 離線產出）
# 格式：{"標準分工關鍵字": ["別名1", "別名2", ...], ...}
_ALIAS_MAPPING_PATH = Path(__file__).parent.parent / "resources" / "alias_mapping.json"


@dataclass
class IntentResult:
    機關: str
    單位: str
    理由: str = ""


class IntentClassifier:
    """
    雙層意圖分類器。

    Layer 1 — Aho-Corasick 快速路徑：
      - load() 從「單位對照表」取出全部分工關鍵字，以正規化後的字串建構 Aho-Corasick 自動機。
      - 若 alias_mapping.json 存在，將口語別名一同加入自動機，擴大比對覆蓋率。
      - classify() 對使用者查詢執行 O(n) 掃描，採「最長匹配優先」解析衝突。

    Layer 2 — LLM 兜底（llm_fallback_classify）：
      - 僅在 classify() 回傳 None（Aho-Corasick 無命中）時觸發，避免常規請求增加延遲。
      - load() 時同步將全量列儲存至 _all_unit_rows，無需再次查詢 DB。
      - 將「單位對照表」所有列以編號清單格式提供給 LLM，請其輸出 JSON 格式的機關/單位。
      - LLM 判斷無關聯或解析失敗時回傳 None。

    所有關鍵字與查詢均先經 normalize_for_matching() 清洗（清除標點、空白），確保比對一致性。
    """

    def __init__(self) -> None:
        self._patterns: list[tuple[str, str, str]] = []  # (keyword, 機關, 單位)
        self._automaton = None
        self._all_unit_rows: list[dict] = []  # 全量列，供 LLM 兜底分類使用
        self._loaded = False

    async def load(self, session: AsyncSession) -> None:
        """Load keyword patterns from database, merged with alias_mapping.json if available."""
        logger.info("從資料庫載入意圖分類器 ...")
        stmt = select(
            單位對照表.c["機關"],
            單位對照表.c["單位"],
            單位對照表.c["分工關鍵字"],
            單位對照表.c["擴充關鍵字"],
        )
        result = await session.execute(stmt)
        rows = result.mappings().all()

        self._patterns = []
        keywords: list[str] = []

        # 儲存全量列供 LLM 兜底分類使用（load 期間 rows 已取出，直接轉存）
        # 同時預先聚合關鍵字供 LLM 參考（按 (機關, 單位) 合併多列，避免同一單位重複出現）
        _unit_rows_map: dict[tuple[str, str], list[str]] = {}
        for r in rows:
            if not (r["機關"] or r["單位"]):
                continue
            key = (r["機關"] or "", r["單位"] or "")
            kw_set = _unit_rows_map.setdefault(key, [])
            seen_in_row: set[str] = set(kw_set)
            if r["分工關鍵字"]:
                for kw in re.split(r"[,;，；、\s]+", r["分工關鍵字"].strip()):
                    kw = kw.strip()
                    if kw and kw not in seen_in_row:
                        kw_set.append(kw)
                        seen_in_row.add(kw)
            if r["擴充關鍵字"]:
                for kw in r["擴充關鍵字"]:
                    kw = kw.strip() if isinstance(kw, str) else ""
                    if kw and kw not in seen_in_row:
                        kw_set.append(kw)
                        seen_in_row.add(kw)

        self._all_unit_rows = [
            {"機關": organ, "單位": unit, "所有關鍵字": kws}
            for (organ, unit), kws in _unit_rows_map.items()
        ]

        # ── Step 1：從 DB 載入標準關鍵字與擴充關鍵字 ────────────────────────────────
        # 同時建立「標準關鍵字 → (機關, 單位)」索引，供別名映射查找
        keyword_to_unit: dict[str, tuple[str, str]] = {}  # normalized_kw → (機關, 單位)

        for row in rows:
            organ = row["機關"] or ""
            unit  = row["單位"] or ""
            
            # 處理原始關鍵字
            raw_keywords = row["分工關鍵字"]
            if raw_keywords:
                for kw in re.split(r"[,;，；、\s]+", raw_keywords.strip()):
                    kw = kw.strip()
                    if not kw: continue
                    normalized_kw = normalize_for_matching(kw)
                    if normalized_kw:
                        self._patterns.append((normalized_kw, organ, unit))
                        keywords.append(normalized_kw)
                        keyword_to_unit[normalized_kw] = (organ, unit)

            # 處理擴充關鍵字
            extended_kws = row["擴充關鍵字"]
            if extended_kws:
                for kw in extended_kws:
                    kw = kw.strip()
                    if not kw: continue
                    normalized_kw = normalize_for_matching(kw)
                    if normalized_kw:
                        self._patterns.append((normalized_kw, organ, unit))
                        keywords.append(normalized_kw)
                        # 注意：若擴充詞與標準詞重複，此處會覆寫，但 (機關, 單位) 相同則無妨
                        if normalized_kw not in keyword_to_unit:
                            keyword_to_unit[normalized_kw] = (organ, unit)

        # ── Step 2：從 alias_mapping.json 載入別名（若存在）─────────────────
        alias_count = 0
        alias_mapping: dict[str, list[str]] = {}

        if _ALIAS_MAPPING_PATH.exists():
            try:
                with open(_ALIAS_MAPPING_PATH, encoding="utf-8") as f:
                    alias_mapping = json.load(f)
                logger.info("別名映射 JSON 載入成功：%d 個 standard_kw", len(alias_mapping))
            except Exception as exc:
                logger.warning("別名映射 JSON 讀取失敗，略過：%s", exc)
                alias_mapping = {}
        else:
            logger.info("別名映射 JSON 不存在（%s），略過別名擴展", _ALIAS_MAPPING_PATH)

        for standard_kw, aliases in alias_mapping.items():
            normalized_standard = normalize_for_matching(standard_kw)
            # 尋找對應的 (機關, 單位)（取第一個匹配的 DB 條目）
            unit_pair = keyword_to_unit.get(normalized_standard)
            if unit_pair is None:
                # 嘗試模糊匹配：DB 中某個關鍵字包含 standard_kw
                for db_kw, pair in keyword_to_unit.items():
                    if normalized_standard in db_kw or db_kw in normalized_standard:
                        unit_pair = pair
                        break
            if unit_pair is None:
                logger.info("別名映射：找不到標準關鍵字 %r 對應的機關/單位，跳過", standard_kw)
                continue

            organ, unit = unit_pair
            for alias in aliases:
                if not isinstance(alias, str):
                    continue
                normalized_alias = normalize_for_matching(alias)
                if not normalized_alias or normalized_alias in keyword_to_unit:
                    continue  # 已存在則跳過，避免重複
                self._patterns.append((normalized_alias, organ, unit))
                keywords.append(normalized_alias)
                alias_count += 1
                logger.debug("別名加入：%r → 機關=%s 單位=%s", alias, organ, unit)

        if _USE_AHO and keywords:
            self._automaton = ahocorasick.AhoCorasick(keywords)

        self._loaded = True
        logger.info(
            "意圖分類器載入完成：DB 關鍵字 %d 個 + 別名 %d 個 = 共 %d 個模式  (aho-corasick=%s)",
            len(keywords) - alias_count, alias_count, len(keywords), _USE_AHO,
        )

    def classify(self, query: str, top_n: int = 1) -> list[IntentResult]:
        """
        Match user query against loaded keyword patterns.
        使用最長匹配優先（longest-match）並依匹配長度降序排列。
        Returns list[IntentResult].
        """
        if not self._loaded or not self._patterns:
            logger.info("意圖分類器尚未載入或無資料，略過比對")
            return []

        normalized_query = normalize_for_matching(query)
        if not normalized_query:
            return []

        results: list[IntentResult] = []
        seen_pairs: set[tuple[str, str]] = set()
        if _USE_AHO and self._automaton is not None:
            matches = self._automaton.find_matches_as_indexes(normalized_query)
            if not matches:
                logger.info("查詢無意圖比對結果（長度=%d）", len(query))
                return []
            
            # 依匹配長度 (end - start) 降序排序
            sorted_matches = sorted(matches, key=lambda m: m[2] - m[1], reverse=True)

            for match in sorted_matches:
                idx = match[0]
                kw, organ, unit = self._patterns[idx]
                pair = (organ, unit)
                if pair not in seen_pairs:
                    results.append(IntentResult(機關=organ, 單位=unit, 理由=f"關鍵字比對到「{kw}」。"))
                    seen_pairs.add(pair)
                if len(results) >= top_n:
                    break
            
            if results:
                logger.info(
                    "意圖比對成功（aho-corasick）：命中 %d 個單位", len(results)
                )
            return results
        else:
            # 降級：線性掃描
            temp_matches: list[tuple[int, str, str, str]] = []  # (length, kw, organ, unit)
            for kw, organ, unit in self._patterns:
                if kw in normalized_query:
                    temp_matches.append((len(kw), kw, organ, unit))
            
            # 依關鍵字長度降序排序
            temp_matches.sort(key=lambda x: x[0], reverse=True)
            
            for length, kw, organ, unit in temp_matches:
                pair = (organ, unit)
                if pair not in seen_pairs:
                    results.append(IntentResult(機關=organ, 單位=unit, 理由=f"關鍵字比對到「{kw}」。"))
                    seen_pairs.add(pair)
                if len(results) >= top_n:
                    break
            
            if results:
                logger.info("意圖比對成功（fallback）：命中 %d 個單位", len(results))
            else:
                logger.info("查詢無意圖比對結果（長度=%d）", len(query))
            return results

    async def llm_fallback_classify(
        self,
        query: str,
        top_n: int = 1,
        interrupt_event: asyncio.Event | None = None,
    ) -> list[IntentResult]:
        """
        LLM 兜底意圖分類：Aho-Corasick 無命中時呼叫。

        將「單位對照表」全量列（含聚合關鍵字）提供給 LLM，採思維鏈 (CoT) 解析。
        """
        if not self._all_unit_rows:
            logger.info("LLM 兜底：無單位資料，略過")
            return []

        # 建立單位與關鍵字聚合列表
        lines = []
        for i, row in enumerate(self._all_unit_rows, start=1):
            organ = row["機關"]
            unit  = row["單位"]
            kws   = "\u3001".join(row["所有關鍵字"])
            lines.append(f"{i}. 機關：{organ}, 單位：{unit}, 分工關鍵字：[{kws}]")
        unit_table = "\n".join(lines)

        prompt = (
            "你是食品輸出主管機關分工分類助理。\n"
            "以下是各機關單位的職責輪廓與對應的分工關鍵字：\n\n"
            f"{unit_table}\n\n"
            "=== 處理與輸出原則 ===\n"
            "1. 推導機制：請先解析提問中的核心行為與實體，尋找與上列關鍵字的同義或包含關係。\n"
            "2. 彈性數量限制：請依據邏輯關聯度挑選負責機關。若有多個機關具備明確責任歸屬，請依關聯度高低列出；若僅有一個，則列出一個；若無任何關聯，請回傳空陣列。不強制湊滿數量，也無須限制最高數量，以精準匹配為唯一原則。\n"
            "3. 嚴格格式分離：請將推導過程放於 <推導分析> 標籤中，最終結果放於 <JSON_Result> 標籤中。\n"
            '4. 每筆 JSON 物件須包含 "理由" 欄位，說明比對到該機關/單位的核心依據（一句話，精簡說明）。\n\n'
            "<推導分析>\n"
            "(你的思考過程)\n"
            "</推導分析>\n"
            "<JSON_Result>\n"
            '[{"機關": "...", "單位": "...", "理由": "..."}, ...]\n'
            "</JSON_Result>\n\n"
            f"<User_Query>{query}</User_Query>"
        )
        try:
            output = await chat_completion(prompt, temperature=0.3, interrupt_event=interrupt_event)
            
            # 從 <JSON_Result> 標籤中擷取內容
            res_match = re.search(r"<JSON_Result>(.*?)</JSON_Result>", output, re.DOTALL)
            json_text = res_match.group(1).strip() if res_match else output
            
            # 尋找陣列結構
            match = re.search(r"\[\s*\{.*\}\s*\]", json_text, re.DOTALL)
            if not match:
                # 嘗試擷取單個 JSON 並包裝成列表
                single_match = re.search(r"\{[^{}]*\}", json_text, re.DOTALL)
                if single_match:
                    data = [json.loads(single_match.group())]
                else:
                    logger.warning("LLM 兜底：無法從輸出中解析 JSON（標籤內內容=%r）", json_text[:200])
                    return []
            else:
                data = json.loads(match.group())
            
            if not isinstance(data, list):
                data = [data]

            results = []
            seen_pairs = set()
            for item in data:
                organ  = str(item.get("機關", "")).strip()
                unit   = str(item.get("單位", "")).strip()
                reason = str(item.get("理由", "")).strip()
                if (organ or unit) and (organ, unit) not in seen_pairs:
                    results.append(IntentResult(機關=organ, 單位=unit, 理由=reason))
                    seen_pairs.add((organ, unit))
                if len(results) >= top_n:
                    break

            if not results:
                logger.info("LLM 兜底：判斷為無相關機關/單位")
            else:
                logger.info("LLM 兜底比對成功：命中 %d 個單位", len(results))
            return results
        except Exception as exc:
            logger.warning("LLM 兜底意圖分類失敗：%s", exc)
            return []


# Module-level singleton
_classifier = IntentClassifier()


async def get_intent_classifier(session: AsyncSession) -> IntentClassifier:
    """Get (and lazily initialize) the intent classifier."""
    if not _classifier._loaded:
        await _classifier.load(session)
    return _classifier


async def reload_intent_classifier(session: AsyncSession) -> IntentClassifier:
    """Force reload the intent classifier from DB."""
    logger.info("強制重新載入意圖分類器 ...")
    await _classifier.load(session)
    return _classifier
