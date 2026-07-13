"""
離線特徵降維工具：從「官方正規詞彙」中提取核心短詞。

功能：
    讀取 DB 中 6000+ 筆官方正規詞彙，透過 Gemma-4 LLM 提取每個術語的核心短詞
    （剝除修飾語，保留 2-4 字核心語意單元）。
    輸出 {短詞 → 官方中文字詞} 的 JSON 映射，供 app/utils/dictionary.py 線上載入。

輸出格式：
    {"牛肉": "牛肉及牛肉製品", "農藥": "農藥殘留容許量", ...}
    鍵 = 核心短詞，值 = 官方中文字詞原文

規則：
    - 使用同步 psycopg（tools/ 規範，非 async）
    - 使用 openai.OpenAI 同步客戶端（tools/ 規範）
    - 不得 import 任何 app/ 模組
    - LLM Server: http://10.166.57.22:40041/v1, model=gemma-4-26B-A4B-it-mtp

反向驗證：
    - 短詞必須是官方原詞的子字串（防 LLM 幻覺）
    - 單字短詞一律丟棄
    - 若一個短詞映射超過 MAX_AMBIGUITY 個不同官方術語，視為過泛用詞，丟棄

CLI 用法：
    python tools/extract_stems.py [--dry-run] [--verbose] [--batch-size 50]
        [--output app/resources/term_mapping.json]
        [--db-host 127.0.0.1] [--db-port 5432] [--db-user postgres]
        [--db-password postgres] [--db-name fes]
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

from db_utils import get_connection, get_schema

# ── 常數 ──────────────────────────────────────────────────────────────────────

LLM_BASE_URL = "http://10.166.57.22:40041/v1"
LLM_MODEL    = "gemma-4-26B-A4B-it-mtp"

# 短詞最大歧義度：若同一短詞對應超過此數的官方術語，視為過泛用，丟棄
MAX_AMBIGUITY = 5

# 過泛用停用短詞（這些詞即使通過驗證也不具鑑別力）
STOPWORDS: set[str] = {
    "食品", "產品", "商品", "物品", "貨品", "貨物",
    "標準", "規定", "規範", "要求", "限制", "辦法",
    "申請", "許可", "認證", "登記", "檢驗", "檢查",
    "進口", "出口", "輸入", "輸出",
    "管理", "業者", "廠商", "機關", "單位",
}

# H3 標題正規表示式（移除修飾詞常見尾綴）
_TRIM_SUFFIX = re.compile(
    r"(及其製品|及製品|及其加工品|加工品|加工食品|加工製品"
    r"|相關製品|衍生物|衍生品|容許量|標準值|限量"
    r"|檢驗基準|作業準則|管理辦法|處理規範"
    r"|其他|等|類)$"
)

_SYSTEM_PROMPT = """\
你是一個食品輸銷術語核心短詞萃取助手。

任務：對每個官方術語，萃取其「核心語意短詞」（2-4 個中文字）。
規則：
1. 核心短詞必須是輸入術語的子字串（直接字元片段，不可翻譯或改寫）。
2. 去除常見修飾語尾綴，例如「及其製品」「容許量」「加工品」「管理辦法」「處理規範」等。
3. 若術語本身已是 2-4 字，直接輸出原詞。
4. 若術語無有意義的核心短詞可提取，輸出空字串 ""。
5. 輸出嚴格的 JSON 陣列，每個元素含 "original" 和 "stem" 兩欄。
6. 不得輸出任何說明文字或 Markdown 格式。

範例輸入：
["牛肉及牛肉製品", "農藥殘留容許量", "有機農產品", "申請許可"]

範例輸出：
[
  {"original": "牛肉及牛肉製品", "stem": "牛肉"},
  {"original": "農藥殘留容許量", "stem": "農藥殘留"},
  {"original": "有機農產品", "stem": "有機農產品"},
  {"original": "申請許可", "stem": ""}
]
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── DB 讀取 ───────────────────────────────────────────────────────────────────

def load_terms_from_db(
    host: str, port: int, user: str, password: str, dbname: str
) -> list[str]:
    """從「官方正規詞彙」資料表讀取全量中文字詞。"""
    schema = get_schema()
    logger.info("連線 DB：%s:%d/%s", host, port, dbname)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT "中文字詞" FROM {schema}."官方正規詞彙" WHERE "中文字詞" IS NOT NULL')
            rows = cur.fetchall()
    terms = [row[0].strip() for row in rows if row[0] and row[0].strip()]
    logger.info("共載入 %d 筆官方中文字詞", len(terms))
    return terms


# ── LLM 萃取 ─────────────────────────────────────────────────────────────────

def _extract_stems_batch(client: OpenAI, batch: list[str]) -> list[dict]:
    """送一批術語至 LLM，取回 [{original, stem}, ...] 列表。"""
    user_content = json.dumps(batch, ensure_ascii=False)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content or ""
        # 容錯：剝除可能的 Markdown code fence
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("JSON 解析失敗（批次 %d 筆）：%s", len(batch), exc)
        return []
    except Exception as exc:
        logger.error("LLM 呼叫失敗：%s", exc)
        return []


# ── 驗證 ──────────────────────────────────────────────────────────────────────

def _validate_stem(original: str, stem: str) -> bool:
    """
    反向驗證：
    1. 短詞必須是原術語的子字串
    2. 長度 >= 2 字
    3. 不在停用詞清單中
    """
    if not stem or len(stem) < 2:
        return False
    if stem not in original:
        return False
    if stem in STOPWORDS:
        return False
    return True


# ── 主流程 ────────────────────────────────────────────────────────────────────

def build_mapping(
    terms: list[str],
    client: OpenAI,
    batch_size: int,
    verbose: bool,
) -> dict[str, str]:
    """
    核心處理流程：
    1. 分批送 LLM 萃取短詞
    2. 反向驗證
    3. 歧義過濾（一個短詞映射太多術語則丟棄）
    回傳 {短詞: 官方中文術語} dict。
    """
    # 先用簡單規則預處理（去除常見尾綴），減少 LLM 工作量
    preprocessed: list[tuple[str, str]] = []  # (original, pre_stem)
    for t in terms:
        pre = _TRIM_SUFFIX.sub("", t).strip()
        preprocessed.append((t, pre))

    # 若預處理後的短詞已符合條件（2-4 字且不是原詞），直接記錄，跳過 LLM
    rule_based: dict[str, str] = {}
    need_llm: list[str] = []

    for original, pre in preprocessed:
        if pre != original and 2 <= len(pre) <= 4 and pre not in STOPWORDS:
            rule_based[pre] = original
            if verbose:
                logger.debug("規則萃取：%r → %r", original, pre)
        else:
            need_llm.append(original)

    logger.info("規則直接萃取 %d 筆，送 LLM 處理 %d 筆", len(rule_based), len(need_llm))

    # LLM 分批萃取
    llm_results: list[dict] = []
    total_batches = (len(need_llm) + batch_size - 1) // batch_size
    for i in range(0, len(need_llm), batch_size):
        batch = need_llm[i: i + batch_size]
        batch_no = i // batch_size + 1
        logger.info("LLM 萃取批次 [%d/%d]，共 %d 筆", batch_no, total_batches, len(batch))
        results = _extract_stems_batch(client, batch)
        llm_results.extend(results)

    # 合併並驗證 LLM 結果
    raw_mapping: dict[str, list[str]] = {}  # stem → [official1, official2, ...]

    # 加入規則萃取結果
    for stem, official in rule_based.items():
        raw_mapping.setdefault(stem, []).append(official)

    # 加入 LLM 萃取結果
    for item in llm_results:
        original = item.get("original", "")
        stem = (item.get("stem") or "").strip()
        if not _validate_stem(original, stem):
            if verbose and stem:
                logger.debug("驗證失敗：original=%r stem=%r", original, stem)
            continue
        raw_mapping.setdefault(stem, []).append(original)

    # 歧義過濾：一個短詞對應太多術語則丟棄
    final_mapping: dict[str, str] = {}
    discarded_ambiguous = 0
    for stem, officials in raw_mapping.items():
        if len(officials) > MAX_AMBIGUITY:
            discarded_ambiguous += 1
            if verbose:
                logger.debug(
                    "歧義過高丟棄：%r 對應 %d 個術語（閾值 %d）",
                    stem, len(officials), MAX_AMBIGUITY,
                )
            continue
        # 取第一個（最先遇到）作為代表
        final_mapping[stem] = officials[0]

    logger.info(
        "最終產出 %d 筆短詞映射（歧義丟棄 %d 筆）",
        len(final_mapping), discarded_ambiguous,
    )
    return final_mapping


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="官方正規詞彙核心短詞萃取工具")
    parser.add_argument("--dry-run", action="store_true", help="預覽模式：不寫入檔案")
    parser.add_argument("--verbose", action="store_true", help="顯示詳細日誌")
    parser.add_argument("--batch-size", type=int, default=50, help="每批 LLM 請求術語數（預設 50）")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "app" / "resources" / "term_mapping.json"),
        help="輸出 JSON 路徑",
    )
    parser.add_argument("--db-host",     default=os.getenv("PG_HOST", "127.0.0.1"))
    parser.add_argument("--db-port",     type=int, default=int(os.getenv("PG_PORT", "5432")))
    parser.add_argument("--db-user",     default=os.getenv("PG_USER", "postgres"))
    parser.add_argument("--db-password", default=os.getenv("PG_PASSWORD", "postgres"))
    parser.add_argument("--db-name",     default=os.getenv("PG_DB", "fes"))
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 讀取 DB
    terms = load_terms_from_db(
        host=args.db_host, port=args.db_port,
        user=args.db_user, password=args.db_password, dbname=args.db_name,
    )
    if not terms:
        logger.error("DB 中無官方正規詞彙資料，中止")
        sys.exit(1)

    # 初始化 LLM 客戶端
    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-required")

    # 建立映射
    mapping = build_mapping(terms, client, args.batch_size, args.verbose)

    if args.dry_run:
        logger.info("--dry-run 模式：顯示前 20 筆結果（不寫入）")
        for stem, official in list(mapping.items())[:20]:
            print(f"  {stem!r:12} → {official!r}")
        logger.info("預覽完成，共 %d 筆映射", len(mapping))
        return

    # 寫入 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info("已寫入：%s  共 %d 筆短詞映射", output_path, len(mapping))


if __name__ == "__main__":
    main()
