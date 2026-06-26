"""
離線別名萃取工具：從「單位對照表」產出 ALIAS_MAPPING JSON。

功能：
    讀取 DB「單位對照表」中每筆記錄的（機關、單位、分工關鍵字），
    透過 Gemma-4 LLM 為每個機關/單位產出常見口語別名、舊名稱、英文縮寫
    及 LLM 翻譯常見變體，輸出 {標準分工關鍵字: [別名, ...]} 的 JSON 映射。

輸出格式：
    {
      "關務署": ["海關", "關稅局", "通關"],
      "食品藥物管理署": ["食藥署", "TFDA"],
      ...
    }
    鍵 = DB 中的某個分工關鍵字（供 intent.py 反查 (機關, 單位) 用）
    值 = 該機關/單位的口語別名清單

規則：
    - 使用同步 psycopg（tools/ 規範，非 async）
    - 使用 openai.OpenAI 同步客戶端（tools/ 規範）
    - 不得 import 任何 app/ 模組
    - LLM Server: http://10.166.57.22:40039/v1, model=gemma-4-26B-A4B-it

驗證：
    - 別名不可與任何原始分工關鍵字完全相同（避免重複）
    - 別名長度 >= 2 字
    - 過濾明顯泛用詞（「主管機關」「有關單位」「相關機關」等）

CLI 用法：
    python tools/extract_aliases.py [--dry-run] [--verbose] [--batch-size 10]
        [--output app/resources/alias_mapping.json]
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

import psycopg
from openai import OpenAI

# ── 常數 ──────────────────────────────────────────────────────────────────────

LLM_BASE_URL = "http://10.166.57.22:40039/v1"
LLM_MODEL    = "gemma-4-26B-A4B-it"

# 過泛用別名停用詞（即使通過長度驗證也不具鑑別力）
ALIAS_STOPWORDS: set[str] = {
    "主管機關", "有關單位", "相關機關", "相關單位", "政府機關", "主管單位",
    "行政機關", "中央主管", "地方主管", "主管部門", "政府部門",
    "機關", "單位", "部門", "機構",
}

_SYSTEM_PROMPT = """\
你是一個台灣政府業務分工關鍵字語意擴充助手，專精於食品輸銷、農業與進出口實務領域。

任務：針對輸入的「分工關鍵字」，請參考其所屬的「機關」與「單位」脈絡，列出台灣民眾、相關業者或在填寫外國問卷時，可能用來指稱該項「業務、領域或主題」的實務慣用語、口語說法、關聯概念或翻譯對應詞彙。

規則：
1. 輸出詞彙必須是實務上確實存在的業務代稱或相關領域說法，不可無中生有。
2. 詞彙長度至少 2 個字。
3. 嚴禁輸出「主管機關」、「相關單位」、「業務負責人」等泛指詞，也不要輸出該機關/單位的名稱。
4. 每個關鍵字最多輸出 6 個擴充詞彙；若該關鍵字具備極高專一性且無合適替代詞，則輸出空陣列。
5. 嚴格輸出純 JSON 陣列格式，不得包含任何說明文字或 Markdown 語法。
6. 輸出陣列的長度必須與輸入陣列嚴格相同，依序一對一對應。

輸入格式範例（JSON 陣列，每個元素含機關、單位、分工關鍵字）：
[
  {"機關": "農業部", "單位": "畜牧司", "分工關鍵字": "家禽產業概況"},
  {"機關": "財政部", "單位": "關務署", "分工關鍵字": "通關"}
]

輸出格式範例（與輸入等長的 JSON 陣列，每個元素僅含 aliases）：
[
  {"aliases": ["養雞業現況", "禽類產銷", "家禽養殖數據"]},
  {"aliases": ["進出口通關", "清關", "報關", "海關查驗"]}
]

說明：
- 輸出陣列的第 i 個元素對應輸入陣列的第 i 個「分工關鍵字」的擴充清單。
- 模型應利用「機關」與「單位」的資訊來準確理解「分工關鍵字」的業務語境，但 aliases 中不應包含關鍵字本身。
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── DB 讀取 ───────────────────────────────────────────────────────────────────

def load_units_from_db(
    host: str, port: int, user: str, password: str, dbname: str
) -> list[dict]:
    """從「單位對照表」讀取全量資料，回傳 list of {機關, 單位, 分工關鍵字}。"""
    dsn = f"host={host} port={port} user={user} password={password} dbname={dbname}"
    logger.info("連線 DB：%s:%d/%s", host, port, dbname)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT "機關", "單位", "分工關鍵字" FROM public."單位對照表" '
                'WHERE "分工關鍵字" IS NOT NULL'
            )
            rows = cur.fetchall()

    units = []
    for organ, unit, kw_raw in rows:
        if not kw_raw or not kw_raw.strip():
            continue
        units.append({
            "機關": (organ or "").strip(),
            "單位": (unit or "").strip(),
            "分工關鍵字": kw_raw.strip(),
        })

    logger.info("共載入 %d 筆單位對照資料", len(units))
    return units


def _expand_keywords(units: list[dict]) -> list[dict]:
    """
    將每筆 DB 記錄的「分工關鍵字」欄位拆分為獨立關鍵字（逗號/分號/空白分隔），
    每個關鍵字獨立成一筆，保留對應的機關與單位資訊。
    同一關鍵字出現多次時保留第一次。
    """
    seen: dict[str, tuple[str, str]] = {}  # keyword → (機關, 單位)
    for unit in units:
        for kw in re.split(r"[,;，；、\s]+", unit["分工關鍵字"]):
            kw = kw.strip()
            if kw and kw not in seen:
                seen[kw] = (unit["機關"], unit["單位"])
    return [
        {"機關": organ, "單位": unit_name, "分工關鍵字": kw}
        for kw, (organ, unit_name) in seen.items()
    ]

def _extract_aliases_batch(client: OpenAI, batch: list[dict]) -> list[list[str]]:
    """
    送一批「分工關鍵字」資料至 LLM，取回每筆關鍵字的別名清單。
    回傳與 batch 等長的 list[list[str]]，依序位置一一對應。
    每個輸入 dict 包含 機關、單位、分工關鍵字（已為獨立關鍵字）。
    """
    user_content = json.dumps(batch, ensure_ascii=False)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or ""
        # 容錯：剝除可能的 Markdown code fence
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        if isinstance(result, list):
            aliases_list: list[list[str]] = []
            for item in result:
                if isinstance(item, dict):
                    raw_aliases = item.get("aliases") or []
                    aliases_list.append([a for a in raw_aliases if isinstance(a, str)])
                elif isinstance(item, list):
                    aliases_list.append([a for a in item if isinstance(a, str)])
                else:
                    aliases_list.append([])
            # 補齊或截斷至 batch 長度（防 LLM 回傳數量不符）
            while len(aliases_list) < len(batch):
                aliases_list.append([])
            return aliases_list[:len(batch)]
        logger.warning("LLM 輸出格式非陣列：%s", type(result))
        return [[] for _ in batch]
    except json.JSONDecodeError as exc:
        logger.warning("JSON 解析失敗（批次 %d 筆）：%s", len(batch), exc)
        return [[] for _ in batch]
    except Exception as exc:
        logger.error("LLM 呼叫失敗：%s", exc)
        return [[] for _ in batch]


# ── 驗證 ──────────────────────────────────────────────────────────────────────

def _validate_alias(alias: str, all_existing_kws: set[str]) -> bool:
    """
    驗證別名是否有效：
    1. 長度 >= 2 字
    2. 不與任何現有分工關鍵字完全相同
    3. 不在泛用停用詞清單中
    """
    alias = alias.strip()
    if len(alias) < 2:
        return False
    if alias in all_existing_kws:
        return False
    if alias in ALIAS_STOPWORDS:
        return False
    return True


# ── 主流程 ────────────────────────────────────────────────────────────────────

def build_alias_mapping(
    units: list[dict],
    client: OpenAI,
    batch_size: int,
    verbose: bool,
) -> dict[str, list[str]]:
    """
    核心處理流程：
    1. 展開「分工關鍵字」為獨立關鍵字（_expand_keywords）
    2. 分批送 LLM 萃取每個關鍵字的別名
    3. 依位置對應（不依賴 LLM 回傳的 key 名稱），避免 LLM 自行選擇代表詞導致對應錯誤
    回傳 {分工關鍵字: [別名1, 別名2, ...]} dict。
    """
    # ── Step 1：展開個別分工關鍵字 ──────────────────────────────────────────
    expanded = _expand_keywords(units)
    logger.info(
        "分工關鍵字展開：%d 筆 DB 記錄 → %d 個獨立關鍵字",
        len(units), len(expanded),
    )

    # 收集所有現有分工關鍵字（供驗證不重複）
    all_raw_kws: set[str] = {e["分工關鍵字"] for e in expanded}
    logger.info("現有分工關鍵字共 %d 個（用於別名去重驗證）", len(all_raw_kws))

    # ── Step 2：LLM 分批萃取（位置對應）────────────────────────────────────
    final_mapping: dict[str, list[str]] = {}
    discarded = 0
    total_batches = (len(expanded) + batch_size - 1) // batch_size

    for i in range(0, len(expanded), batch_size):
        batch = expanded[i: i + batch_size]
        batch_no = i // batch_size + 1
        logger.info(
            "LLM 別名萃取批次 [%d/%d]，共 %d 個關鍵字",
            batch_no, total_batches, len(batch),
        )
        aliases_list = _extract_aliases_batch(client, batch)

        # 依位置一一對應：batch[j]["分工關鍵字"] → aliases_list[j]
        for item, raw_aliases in zip(batch, aliases_list):
            kw = item["分工關鍵字"]
            valid_aliases: list[str] = []
            for alias in raw_aliases:
                alias = alias.strip()
                if _validate_alias(alias, all_raw_kws):
                    valid_aliases.append(alias)
                    if verbose:
                        logger.debug("有效別名：%r → 關鍵字=%r", alias, kw)
                else:
                    discarded += 1
                    if verbose:
                        logger.debug("丟棄別名：%r（未通過驗證）", alias)

            if kw in final_mapping:
                existing = set(final_mapping[kw])
                final_mapping[kw].extend(a for a in valid_aliases if a not in existing)
            else:
                final_mapping[kw] = valid_aliases

    logger.info(
        "別名萃取完成：%d 個關鍵字，有效別名共 %d 個，丟棄 %d 個",
        len(final_mapping),
        sum(len(v) for v in final_mapping.values()),
        discarded,
    )
    return {k: v for k, v in final_mapping.items() if v}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="單位對照表別名萃取工具（產出 alias_mapping.json）")
    parser.add_argument("--dry-run",    action="store_true", help="預覽模式：不寫入檔案")
    parser.add_argument("--verbose",    action="store_true", help="顯示詳細日誌")
    parser.add_argument("--batch-size", type=int, default=10, help="每批 LLM 請求筆數（預設 10）")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "app" / "resources" / "alias_mapping.json"),
        help="輸出 JSON 路徑（預設 app/resources/alias_mapping.json）",
    )
    parser.add_argument("--db-host",     default=os.getenv("PG_HOST",     "127.0.0.1"))
    parser.add_argument("--db-port",     type=int, default=int(os.getenv("PG_PORT", "5432")))
    parser.add_argument("--db-user",     default=os.getenv("PG_USER",     "postgres"))
    parser.add_argument("--db-password", default=os.getenv("PG_PASSWORD", "postgres"))
    parser.add_argument("--db-name",     default=os.getenv("PG_DB",       "fes"))
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 讀取 DB
    units = load_units_from_db(
        host=args.db_host, port=args.db_port,
        user=args.db_user, password=args.db_password, dbname=args.db_name,
    )
    if not units:
        logger.error("DB 中無單位對照表資料，中止")
        sys.exit(1)

    # 初始化 LLM 客戶端
    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-required")

    # 建立別名映射
    mapping = build_alias_mapping(units, client, args.batch_size, args.verbose)

    if args.dry_run:
        logger.info("--dry-run 模式：顯示所有結果（不寫入）")
        for std_kw, aliases in mapping.items():
            print(f"  {std_kw!r:20} → {aliases}")
        logger.info("預覽完成，共 %d 個 standard_kw，%d 個別名",
                    len(mapping), sum(len(v) for v in mapping.values()))
        return

    # 寫入 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info("已寫入：%s  共 %d 個 standard_kw，%d 個別名",
                output_path, len(mapping), sum(len(v) for v in mapping.values()))


if __name__ == "__main__":
    main()
