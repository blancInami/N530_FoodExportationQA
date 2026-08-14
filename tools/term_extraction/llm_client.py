"""
模組二（Part A）：LLM 客戶端封裝

功能：
1. 建立指向本地 Gemma 4 模型的 OpenAI-compatible 客戶端
2. 封裝「雙語段落 → 專有名詞 JSON」的推論呼叫
3. 提供 JSON 解析容錯機制，避免 LLM 輸出格式不穩定導致程式中斷

LLM 連線資訊：
    HOST : 10.166.57.22
    PORT : 40041
    MODEL: gemma-4-26B-A4B-it-mtp
"""

import json
import logging
import re

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── 常數設定 ─────────────────────────────────────────────────────────────────

LLM_BASE_URL = "http://10.166.57.22:40041/v1"
LLM_MODEL    = "gemma-4-26B-A4B-it-mtp"

# 系統提示詞：明確限制模型僅能擷取原文中確實存在的詞彙，禁止捏造或推測
_SYSTEM_PROMPT = """\
你是一位專門處理食品輸銷領域的嚴謹雙語專有名詞擷取助手。

【領域範圍】
僅擷取與「食品輸出、銷售、貿易及相關法規」有關的詞彙，包含：
- 食品類別與品名（如：農產品、加工食品、有機食品）
- 食品安全與衛生法規術語（如：殘留農藥、食品添加物、衛生標準）
- 輸出入程序與文件（如：檢疫證明、產地證明、輸出許可）
- 主管機關與認證機構名稱（如：食品藥物管理署、動植物防疫檢疫署）
- 輸出目的國相關法規或標準（如：FDA、EU Regulation）
與食品輸銷無直接關聯的一般性詞彙（如純行政程序、一般法律術語）不得納入。

【擷取規則】
1. 僅能擷取在以下中文段落與英文段落中「確實出現」的詞彙。
2. 嚴禁捏造、推測或翻譯原文中不存在的詞彙。
3. 擷取的詞彙必須同時在中文段落和英文段落中皆可找到對應詞彙（可為翻譯對應）。
4. 通用常見詞（如「公司」、「document」、「申請」、「application」）不應納入。
5. 必須從段落內容中識別負責該問題的單位名稱。
6. 必須嚴格輸出純 JSON，不得包含任何說明文字或 Markdown 格式。

【輸出格式】
輸出一個 JSON 陣列，每個元素的結構如下：
[
  {
    "chinese_term": "中文專有名詞",
    "english_term": "English Term",
    "responsible_unit": "負責單位名稱"
  },
  ...
]

若無符合食品輸銷領域條件的詞彙，輸出空陣列：[]
"""


# ── 客戶端建立 ────────────────────────────────────────────────────────────────

def create_openai_client() -> OpenAI:
    """
    建立指向本地 Gemma 4 推論伺服器的 OpenAI-compatible 客戶端。

    本地部署的模型不需要 API 金鑰，故 api_key 填入任意非空字串即可。

    Returns:
        openai.OpenAI 實例，base_url 已指向本地伺服器。
    """
    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key="not-needed",  # 本地模型不驗證金鑰，填入佔位字串
    )
    logger.debug("OpenAI client 已建立：base_url=%s  model=%s", LLM_BASE_URL, LLM_MODEL)
    return client


# ── HTML 雙語欄位專用 System Prompt ───────────────────────────────────────────

_HTML_SYSTEM_PROMPT = """\
你是一位專門處理食品輸銷領域的嚴謹雙語專有名詞擷取助手。

【輸入格式說明】
輸入包含食品輸銷問卷的「題目」與「回覆」HTML 內容。
每個部分同時包含英文與中文，兩種語言通常以 <br> 或 <p> 等 HTML 標籤分隔。

請直接從 HTML 原文中識別中英對應的專有名詞，無需關注 HTML 標籤結構。

請特別留意並分析「題目」內容中的負責單位線索，常見模式包括：
- 只在「題目」HTML 結尾**可能**會出現指名負責單位。
- 位於 HTML 標籤內，如：「請<strong><u>防檢局</u></strong>填答」、「請<strong>防檢署</strong>填答」
- 括號標註，如：「<strong>（請食品組、漁業署填答）</strong>」、「(請動植物防疫檢疫署填答)」
- 排除英文縮寫（如 FDA、EU）或一般詞彙（如 company、document）等非單位名稱。
若無法從內容判斷負責單位，則填入空字串。

【領域範圍】
僅擷取與「食品輸出、銷售、貿易及相關法規」有關的詞彙，包含：
- 食品類別與品名（如：農產品、加工食品、有機食品）
- 食品安全與衛生法規術語（如：殘留農藥、食品添加物、衛生標準）
- 輸出入程序與文件（如：檢疫證明、產地證明、輸出許可）
- 主管機關與認證機構名稱（如：食品藥物管理署、動植物防疫檢疫署）
- 輸出目的國相關法規或標準（如：FDA、EU Regulation、Directive 96/93/EC）
與食品輸銷無直接關聯的一般性詞彙不得納入。

【擷取規則】
1. 僅能擷取在輸入 HTML 中「確實出現」的詞彙，禁止捏造或推測。
2. 中英文詞彙必須在輸入內容內可找到對應關係（翻譯對應即可）。
3. 通用常見詞（如「公司」、「document」、「申請」、「application」）不應納入。
4. 必須識別負責單位：優先尋找「請XXX填答」或括號內的單位指名（常見於 <strong> 或 <u> 標籤中）。若無法從內容判斷，則填入空字串。
5. 必須嚴格輸出純 JSON，不得包含任何說明文字或 Markdown 格式。

【輸出格式】
[
  {
    "chinese_term": "中文專有名詞",
    "english_term": "English Term",
    "responsible_unit": "負責單位名稱"
  },
  ...
]

若無符合食品輸銷領域條件的詞彙，輸出空陣列：[]
"""


# ── 核心推論函式 ──────────────────────────────────────────────────────────────

def extract_terms_from_html(
    client: OpenAI,
    html_content: str,
) -> list[dict]:
    """
    將單筆 HTML 欄位值直接送入 Gemma 4，擷取雙語專有名詞對照。

    適用於「問卷題目檔」的「題目」或「回覆」欄位，
    內容為同時含中英文的 HTML 格式文字。

    Args:
        client      : 由 create_openai_client() 建立的客戶端
        html_content: HTML 格式字串（含中英混合文字）

    Returns:
        list[dict]，每個 dict 含 'chinese_term' 與 'english_term'。
        若推論失敗或無符合詞彙，回傳空列表。
    """
    user_prompt = (
        "【HTML 欄位內容】\n"
        f"{html_content.strip()}\n\n"
        "請依照規則從以上 HTML 內容中擷取食品輸銷雙語專有名詞，輸出純 JSON 陣列。"
    )

    logger.debug("LLM HTML 推論：HTML 長度=%d", len(html_content))

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _HTML_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            max_tokens=2048,
        )
    except Exception as exc:
        logger.warning("LLM 呼叫失敗，跳過此 HTML 欄位：%s", exc)
        return []

    raw_content: str = response.choices[0].message.content or ""
    logger.debug("LLM 原始輸出（前 300 字）：%s", raw_content[:300])
    return _parse_json_response(raw_content)


def extract_terms_from_qa(
    client: OpenAI,
    title_html: str,
    reply_html: str,
) -> list[dict]:
    """
    同時將問卷的「題目」與「回覆」送入 LLM，擷取雙語專有名詞與負責單位。

    Args:
        client: OpenAI-compatible 客戶端
        title_html: 題目欄位 HTML
        reply_html: 回覆欄位 HTML

    Returns:
        list[dict]，含 'chinese_term', 'english_term', 'responsible_unit'
    """
    user_prompt = (
        "【問卷題目內容】\n"
        f"{title_html.strip()}\n\n"
        "【問卷回覆內容】\n"
        f"{reply_html.strip()}\n\n"
        "請依照規則從以上內容中擷取食品輸銷雙語專有名詞並識別負責單位，輸出純 JSON 陣列。"
    )

    logger.debug("LLM QA 聯合推論：題目長度=%d  回覆長度=%d", len(title_html), len(reply_html))

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _HTML_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            max_tokens=3000,
        )
    except Exception as exc:
        logger.warning("LLM 聯合推論失敗：%s", exc)
        return []

    raw_content: str = response.choices[0].message.content or ""
    return _parse_json_response(raw_content)


def extract_terms_from_chunk(
    client: OpenAI,
    zh_chunk: str,
    en_chunk: str,
) -> list[dict]:
    """
    將一對中英文段落送入 Gemma 4，擷取雙語專有名詞對照。

    防幻覺設計：
    - temperature=0：最大化輸出確定性，降低模型隨機生成不存在詞彙的機率
    - System prompt 明確禁止捏造
    - 輸出解析容錯：LLM 格式不穩定時，嘗試從輸出中正則提取 JSON 區塊

    Args:
        client   : 由 create_openai_client() 建立的客戶端
        zh_chunk : 中文文本段落（已切塊）
        en_chunk : 英文文本段落（已切塊）

    Returns:
        list[dict]，每個 dict 含 'chinese_term' 與 'english_term'。
        若推論失敗或無符合詞彙，回傳空列表。
    """
    # 組合使用者提示，明確標示中英文段落邊界
    user_prompt = (
        "【中文段落】\n"
        f"{zh_chunk.strip()}\n\n"
        "【英文段落】\n"
        f"{en_chunk.strip()}\n\n"
        "請依照規則擷取雙語專有名詞，輸出純 JSON 陣列。"
    )

    logger.debug(
        "LLM 推論：中文段落長度=%d  英文段落長度=%d",
        len(zh_chunk), len(en_chunk),
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,      # 確定性輸出，降低幻覺機率
            max_tokens=2048,    # 單次最大 token 數，避免過長輸出
        )
    except Exception as exc:
        # 網路錯誤、逾時或伺服器錯誤：記錄後跳過此 chunk
        logger.warning("LLM 呼叫失敗，跳過此 chunk：%s", exc)
        return []

    # 取得模型回應文字
    raw_content: str = response.choices[0].message.content or ""
    logger.debug("LLM 原始輸出（前 300 字）：%s", raw_content[:300])

    return _parse_json_response(raw_content)


def _parse_json_response(raw: str) -> list[dict]:
    """
    解析 LLM 回傳的 JSON 字串，具備三層容錯機制：
    1. 直接解析整個回應（理想情況）
    2. 用正則從回應中提取 [...] 區塊後解析（LLM 多了說明文字時）
    3. 上述皆失敗則回傳空列表並記錄 warning

    Args:
        raw: LLM 回傳的原始文字

    Returns:
        解析後的 list[dict]，每個 dict 含 'chinese_term' 與 'english_term'。
        格式不合規的項目會被過濾。
    """
    # 第一層：直接嘗試解析整個輸出
    try:
        parsed = json.loads(raw.strip())
        return _validate_term_list(parsed)
    except json.JSONDecodeError:
        pass

    # 第二層：嘗試用正則提取第一個 [...] 區塊
    # 使用非貪婪模式找出最外層 [...] 的完整區塊
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return _validate_term_list(parsed)
        except json.JSONDecodeError:
            pass

    # 第三層：嘗試提取以 ``` 包裹的 JSON 程式碼區塊（部分模型習慣輸出 ```json ... ```）
    code_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if code_match:
        try:
            parsed = json.loads(code_match.group(1).strip())
            return _validate_term_list(parsed)
        except json.JSONDecodeError:
            pass

    # 所有解析方式均失敗
    logger.warning("無法解析 LLM 輸出為有效 JSON，跳過此 chunk。原始輸出前 200 字：%s", raw[:200])
    return []


def _validate_term_list(parsed: object) -> list[dict]:
    """
    驗證解析後的物件是否為合法的詞彙列表格式，
    過濾掉缺少必要欄位或欄位值為空的項目。

    Args:
        parsed: json.loads 的解析結果

    Returns:
        合法的 list[dict] 清單
    """
    if not isinstance(parsed, list):
        logger.warning("LLM 輸出的根節點不是 JSON 陣列，跳過。類型：%s", type(parsed).__name__)
        return []

    valid: list[dict] = []
    for item in parsed:
        # 每個項目必須是 dict，且含有非空的 chinese_term 與 english_term
        if (
            isinstance(item, dict)
            and isinstance(item.get("chinese_term"), str)
            and isinstance(item.get("english_term"), str)
            and item["chinese_term"].strip()
            and item["english_term"].strip()
        ):
            valid.append({
                "chinese_term": item["chinese_term"].strip(),
                "english_term": item["english_term"].strip(),
                "responsible_unit": str(item.get("responsible_unit", "")).strip(),
            })
        else:
            logger.debug("過濾掉格式不合規的項目：%s", item)

    return valid
