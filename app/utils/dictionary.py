"""
Official mandatory dictionary for specialized terminology translation.
Loads Chinese → English official terms from the database table「官方正規詞彙」.

Strategy: DictionaryMatcher uses Aho-Corasick for O(n) substring scan.
Only terms that actually appear in the input text are injected into the LLM prompt,
keeping prompt size manageable even with 6000+ entries in the database.

Enhancement: Optionally loads app/resources/term_mapping.json (produced by
tools/extract_stems.py) to register core short-terms in the automaton.
Short-term hits are reverse-mapped to their official full terms before output,
so callers always receive {official_zh: official_en} regardless of whether
the match was triggered by the full term or a short-term alias.
"""
import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import 官方正規詞彙
from app.utils import normalize_for_matching

try:
    import ahocorasick_rs as ahocorasick
    _USE_AHO = True
except ImportError:
    _USE_AHO = False

logger = logging.getLogger(__name__)

# 短詞映射 JSON 的預設路徑（由 tools/extract_stems.py 產出）
_TERM_MAPPING_PATH = Path(__file__).parent.parent / "resources" / "term_mapping.json"


async def load_dictionary(session: AsyncSession) -> dict[str, str]:
    """
    從資料庫「官方正規詞彙」資料表載入官方術語對照字典（全量）。

    Returns:
        dict[str, str]：中文字詞 → 英文字詞 的對照表。
        若資料表為空或查詢失敗，回傳空字典。
    """
    try:
        stmt = select(
            官方正規詞彙.c["中文字詞"],
            官方正規詞彙.c["英文字詞"],
        )
        result = await session.execute(stmt)
        rows = result.fetchall()
        return {
            row[0]: row[1]
            for row in rows
            if row[0] and row[1]
        }
    except Exception as exc:
        logger.error("載入官方正規詞彙失敗，回傳空字典：%s", exc)
        return {}


def build_dictionary_xml(items: dict[str, str]) -> str:
    """Build the <Mandatory_Dictionary> XML block for LLM prompts."""
    if not items:
        return "<Mandatory_Dictionary></Mandatory_Dictionary>"
    entries = []
    for zh, en in items.items():
        entries.append(f"  <term><zh>{zh}</zh><en>{en}</en></term>")
    return "<Mandatory_Dictionary>\n" + "\n".join(entries) + "\n</Mandatory_Dictionary>"


# ── DictionaryMatcher ────────────────────────────────────────────────────────

class DictionaryMatcher:
    """
    Aho-Corasick 字典比對器（含短詞別名支援）。

    核心功能：
    - 對照「官方正規詞彙」資料表建立 Aho-Corasick 自動機，O(n) 掃描文本。
    - 若 app/resources/term_mapping.json 存在，將短詞（核心語意詞幹）
      同時加入自動機；命中短詞時反查至官方完整術語後輸出，
      呼叫端永遠收到 {官方中文 → 官方英文} 格式。

    Phase 1c：filter_by_text(query) — 正常掃描，含短詞觸發
    Phase 3 ：filter_by_text_grounded(query, context) — 額外防護，
              若術語**僅**被短詞觸發且官方完整詞彙在 context 中不存在，
              則從結果中剔除，防止泛用短詞引入字典幻覺。

    使用模式：
        matcher = await get_dictionary_matcher(session)
        # Phase 1c
        filtered = matcher.filter_by_text(chinese_query)
        # Phase 3
        filtered = matcher.filter_by_text_grounded(chinese_query, raw_context)
        xml = build_dictionary_xml(filtered)
    """

    def __init__(self) -> None:
        self._zh_terms: list[str] = []            # automaton 模式列表（含短詞）
        self._en_terms: list[str] = []            # 對應英文（短詞佔位為空字串）
        self._automaton = None
        self._fallback_dict: dict[str, str] = {}  # 無 aho 時的降級字典（官方原詞）
        self._stem_to_official: dict[str, str] = {}  # 短詞 → 官方中文字詞
        self._is_stem_idx: set[int] = set()        # 在 _zh_terms 中屬於短詞的索引
        self._loaded = False

    async def load(self, session: AsyncSession) -> None:
        """
        從資料庫載入官方詞彙，並嘗試讀取短詞映射 JSON，一併建構 Aho-Corasick 自動機。
        所有模式均先經 normalize_for_matching() 清洗，確保與查詢文本的清洗一致。
        """
        logger.info("載入 DictionaryMatcher ...")
        items = await load_dictionary(session)

        # ── Step 1：官方詞彙 ─────────────────────────────────────────────────
        normalized_items: dict[str, str] = {}
        for zh, en in items.items():
            nzh = normalize_for_matching(zh)
            if nzh:
                normalized_items[nzh] = en

        self._zh_terms = list(normalized_items.keys())
        self._en_terms = list(normalized_items.values())
        self._fallback_dict = normalized_items
        self._stem_to_official = {}
        self._is_stem_idx = set()

        # 同時保留原始中文 → 英文的映射（未正規化），供 filter 回傳原始官方詞
        self._raw_zh_to_en: dict[str, str] = {
            normalize_for_matching(zh): en for zh, en in items.items() if normalize_for_matching(zh)
        }
        # 正規化後詞 → 官方原始中文（用於 XML 回傳時保留原始寫法）
        self._normalized_to_raw: dict[str, str] = {
            normalize_for_matching(zh): zh for zh in items if normalize_for_matching(zh)
        }

        official_count = len(self._zh_terms)

        # ── Step 2：讀取短詞映射 JSON ────────────────────────────────────────
        stem_count = 0
        if _TERM_MAPPING_PATH.exists():
            try:
                with open(_TERM_MAPPING_PATH, encoding="utf-8") as f:
                    raw_mapping: dict[str, str] = json.load(f)

                for stem, official_zh in raw_mapping.items():
                    n_stem = normalize_for_matching(stem)
                    n_official = normalize_for_matching(official_zh)
                    if not n_stem or not n_official:
                        continue
                    # 跳過與官方原詞完全相同的短詞（已在自動機中）
                    if n_stem in normalized_items:
                        continue
                    # 短詞加入 _zh_terms，英文佔位空字串（命中時透過反查取得）
                    idx = len(self._zh_terms)
                    self._zh_terms.append(n_stem)
                    self._en_terms.append("")
                    self._is_stem_idx.add(idx)
                    # 短詞 → 官方原始中文（反查用）
                    self._stem_to_official[n_stem] = official_zh
                    stem_count += 1

                logger.info(
                    "短詞映射 JSON 載入完成：%d 筆短詞（丟棄重複 %d 筆）",
                    stem_count, len(raw_mapping) - stem_count,
                )
            except Exception as exc:
                logger.warning("短詞映射 JSON 讀取失敗，略過：%s", exc)
        else:
            logger.debug("短詞映射 JSON 不存在（%s），略過短詞擴展", _TERM_MAPPING_PATH)

        # ── Step 3：建構 Aho-Corasick 自動機 ─────────────────────────────────
        if _USE_AHO and self._zh_terms:
            self._automaton = ahocorasick.AhoCorasick(self._zh_terms)

        self._loaded = True
        logger.info(
            "DictionaryMatcher 載入完成：官方詞彙 %d 筆 + 短詞 %d 筆 = 共 %d 個自動機模式  (aho-corasick=%s)",
            official_count, stem_count, len(self._zh_terms), _USE_AHO,
        )

    def _resolve_match(self, pattern_idx: int) -> tuple[str, str] | None:
        """
        將自動機匹配索引解析為 (官方原始中文, 英文) 對。
        若為短詞索引，透過 _stem_to_official 反查官方術語，再取對應英文。
        回傳 None 表示無法解析（反查失敗）。
        """
        n_zh = self._zh_terms[pattern_idx]

        if pattern_idx in self._is_stem_idx:
            # 短詞 → 反查官方中文 → 查英文
            official_zh = self._stem_to_official.get(n_zh)
            if not official_zh:
                return None
            raw_zh = official_zh  # 短詞映射值為原始官方中文
            n_official = normalize_for_matching(official_zh)
            en = self._raw_zh_to_en.get(n_official, "")
            return (raw_zh, en)
        else:
            # 官方原詞 → 直接取英文，回傳原始寫法
            raw_zh = self._normalized_to_raw.get(n_zh, n_zh)
            en = self._en_terms[pattern_idx]
            return (raw_zh, en)

    def _scan(self, combined: str) -> dict[str, str]:
        """對 combined 文本執行掃描，回傳 {官方原始中文: 英文}。"""
        matched: dict[str, str] = {}

        if _USE_AHO and self._automaton is not None:
            for pattern_idx, _start, _end in self._automaton.find_matches_as_indexes(combined):
                pair = self._resolve_match(pattern_idx)
                if pair and pair[0] not in matched:
                    matched[pair[0]] = pair[1]
        else:
            # 降級：線性掃描（僅官方原詞，不含短詞）
            for n_zh, en in self._fallback_dict.items():
                if n_zh in combined:
                    raw_zh = self._normalized_to_raw.get(n_zh, n_zh)
                    matched[raw_zh] = en

        return matched

    def filter_by_text(self, *texts: str) -> dict[str, str]:
        """
        掃描一或多段文本，回傳其中實際出現的中英對照子集。
        Phase 1c 使用此方法（無 context grounding 防護）。

        Args:
            *texts: 任意數量的文本字串（如 chinese_query、raw_context）

        Returns:
            dict[str, str]：{官方中文字詞 → 英文字詞}，命中短詞時已反查至完整官方術語。
        """
        if not self._loaded or not self._zh_terms:
            return {}

        combined = normalize_for_matching("\n".join(t for t in texts if t))
        if not combined:
            return {}

        matched = self._scan(combined)

        logger.debug(
            "DictionaryMatcher.filter_by_text：掃描 %d 字元  命中 %d / %d 個模式",
            len(combined), len(matched), len(self._zh_terms),
        )
        return matched

    def filter_by_text_grounded(self, query: str, context: str) -> dict[str, str]:
        """
        Phase 3 專用：帶 context grounding 防護的掃描方法。

        在 filter_by_text(query, context) 的基礎上，額外剔除以下術語：
        - 該術語的官方完整詞彙**僅**因短詞命中 query，且在 context 中完全不存在。

        目的：防止口語化短詞（如「牛肉」）在 raw_context 無相關資料時，
        仍將「牛肉及牛肉製品」注入 LLM 字典，導致幻覺性術語植入。

        Args:
            query  : chinese_query（使用者查詢，已翻譯為中文）
            context: retrieval_result.raw_context（檢索回來的問卷/文獻上下文）

        Returns:
            dict[str, str]：通過 grounding 驗證的 {官方中文 → 英文} 子集。
        """
        if not self._loaded or not self._zh_terms:
            return {}

        n_query   = normalize_for_matching(query)
        n_context = normalize_for_matching(context)
        combined  = n_query + "\n" + n_context if n_context else n_query

        if not combined.strip():
            return {}

        # 掃描 query + context 合并文本
        all_matched = self._scan(combined)

        if not self._is_stem_idx:
            # 無短詞模式，grounding 無需額外篩選
            return all_matched

        # 掃描純 context，取得 context 中獨立命中的術語
        context_matched: set[str] = set()
        if n_context:
            for raw_zh in self._scan(n_context):
                context_matched.add(raw_zh)

        # 掃描純 query，取得 query 中命中的術語
        query_matched: set[str] = set()
        if n_query:
            for raw_zh in self._scan(n_query):
                query_matched.add(raw_zh)

        grounded: dict[str, str] = {}
        for raw_zh, en in all_matched.items():
            n_official = normalize_for_matching(raw_zh)
            # 若術語在 context 中有命中 → 保留
            if raw_zh in context_matched:
                grounded[raw_zh] = en
                continue
            # 若官方完整詞彙直接出現在 query 中 → 保留
            if n_official in n_query:
                grounded[raw_zh] = en
                continue
            # 其餘：僅由 query 中的短詞觸發，且 context 無佐證 → 剔除
            logger.debug(
                "grounding 剔除：%r（僅短詞觸發，context 無佐證）", raw_zh
            )

        logger.debug(
            "DictionaryMatcher.filter_by_text_grounded：命中 %d → grounding 後 %d",
            len(all_matched), len(grounded),
        )
        return grounded

    def filter_by_english_text(self, text: str) -> dict[str, str]:
        """
        對英文文本執行術語比對，回傳英文術語出現在輸入中的中英對照子集。
        供 Phase 1a 翻譯前術語約束使用，以官方正規詞彙限制 LLM 翻譯結果。

        Args:
            text: 英文輸入文本（case-insensitive 比對）

        Returns:
            dict[str, str]：{官方中文字詞 → 英文字詞}
        """
        if not self._loaded or not self._raw_zh_to_en:
            return {}

        text_lower = text.lower()
        matched: dict[str, str] = {}

        for n_zh, en in self._raw_zh_to_en.items():
            if not en:
                continue
            if en.lower() in text_lower:
                raw_zh = self._normalized_to_raw.get(n_zh, n_zh)
                matched[raw_zh] = en

        logger.debug(
            "DictionaryMatcher.filter_by_english_text：掃描 %d 字元  命中 %d 筆術語",
            len(text), len(matched),
        )
        return matched


# ── 模組級單例與工廠函式（與 intent.py 相同模式）──────────────────────────────

_matcher = DictionaryMatcher()


async def get_dictionary_matcher(session: AsyncSession) -> DictionaryMatcher:
    """取得（並懶載入）DictionaryMatcher 單例。"""
    if not _matcher._loaded:
        await _matcher.load(session)
    return _matcher


async def reload_dictionary_matcher(session: AsyncSession) -> DictionaryMatcher:
    """強制從資料庫重新載入 DictionaryMatcher。"""
    logger.info("強制重新載入 DictionaryMatcher ...")
    await _matcher.load(session)
    return _matcher
