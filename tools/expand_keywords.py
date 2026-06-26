"""
Offline tool: Expand "分工關鍵字" for "單位對照表" using LLM.
This tool reads (機關, 單位, 分工關鍵字), calls LLM to generate 5-8 expanded keyword,
and stores them back into the "擴充關鍵字" (TEXT[]) column.

Requirements:
- PostgreSQL connection (via .env or default)
- LLM Server running (Gemma 4 requested by system prompt rules)
"""
import os
import json
import logging
import re
import psycopg
from dotenv import load_dotenv
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Database config
PG_DSN = f"host={os.getenv('PG_HOST', '127.0.0.1')} port={os.getenv('PG_PORT', '5432')} user={os.getenv('PG_USER', 'postgres')} password={os.getenv('PG_PASSWORD', 'postgres')} dbname={os.getenv('PG_DB', 'fes')}"

# LLM config
LLM_URL = f"http://{os.getenv('LLM_HOST', '10.166.57.22')}:{os.getenv('LLM_PORT', '40036')}/v1/chat/completions"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss-20b")

SYSTEM_PROMPT = """
任務：你是一位具備台灣食品、農漁業、動植物防疫檢驗領域專業知識的資料特徵工程助理。
目標：請以提供的「既有分工關鍵字」為絕對核心，並參考其所屬的「主管機關與單位」作為情境邊界，進行實務同義詞與延伸情境詞的擴充，以優化意圖分類系統的辨識率。

【目標機關與單位】
機關：{agency_name}
單位：{unit_name}

【既有分工關鍵字】
{existing_keyword}

【思考與擴充原則】
1. 語言與慣用語絕對限制：所有生成的擴充詞彙必須為標準的正體中文，並嚴格遵守台灣地區公部門與產業實務的慣用語。絕對禁止包含簡體中文或非台灣本地的用詞。若為國際通用之專有名稱縮寫（如 MRL、HACCP 等），可直接保留純英文字母，不可附加括號解釋。
2. 核心基準定錨：擴充詞彙必須從本次提供的「既有分工關鍵字」之具體概念出發，嚴禁直接套用該機關的廣義通用職責。例如：同為「食藥署」，若既有關鍵字為「食品安全法規」，擴充詞應聚焦法規條文層次；若既有關鍵字為「食品中毒」，則應聚焦病原、通報與流行病學調查。
3. 領域特徵綁定：禁止生成「稽查」、「管理」、「檢驗」、「計畫」、「法規」等缺乏上下文的孤立詞彙。擴充詞必須自帶領域特徵（例如：將「管理」具體化為「飼料流向管理」，將「檢驗」具體化為「微生物檢驗方法」）。
4. 實務情境轉換：考量提問者可能使用的白話文或實務別稱。例如將「制定最大殘留容許限量」擴充為「殘留標準」、「容許量評估」。
5. 數量限制：請精準產出 5 至 8 個擴充關鍵字。

=== 輸出格式要求 ===
請嚴格遵循以下兩段式輸出：

<思考過程>
請在此簡要分析本次「既有分工關鍵字」的微觀業務屬性，並說明將如何確保擴充詞彙符合台灣在地慣用語，且避免與該機關的其他業務領域產生重疊。
</思考過程>
<JSON_Result>
["擴充詞1", "擴充詞2", "擴充詞3", ...]
</JSON_Result>
"""

def call_llm(agency, unit, keyword):
    prompt = SYSTEM_PROMPT.format(
        agency_name=agency,
        unit_name=unit,
        existing_keyword=keyword
    )
    
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(LLM_URL, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Extract JSON array
        match = re.search(r"\[\s*.*?\s*\]", content, re.DOTALL)
        if match:
            return json.loads(match.group()),keyword
        else:
            logger.error(f"Failed to parse LLM output for {agency}-{unit}: {content[:100]}")
            return None,keyword
    except Exception as e:
        logger.error(f"LLM call failed for {agency}-{unit}: {e}")
        return None,keyword

def main():
    logger.info("Starting keyword expansion...")
    
    try:
        with psycopg.connect(PG_DSN) as conn:
            with conn.cursor() as cur:
                # 1. Fetch all rows
                cur.execute('SELECT "機關", "單位", "分工關鍵字" FROM public."單位對照表"')
                rows = cur.fetchall()
                logger.info(f"Found {len(rows)} units to process.")
                
                for agency, unit, keyword in rows:
                    logger.info(f"Processing {agency} - {unit} with existing keyword: {keyword}")
                    expanded,keyword = call_llm(agency or "", unit or "", keyword or "")
                    
                    if expanded and isinstance(expanded, list):
                        logger.info(f"Generated {len(expanded)} keywords: {expanded}")
                        # 2. Update the row
                        # PostgreSQL requires array literal formatted like {'item1', 'item2'} or using psycopg's array adaptation
                        cur.execute(
                            'UPDATE public."單位對照表" SET "擴充關鍵字" = %s WHERE "機關" IS NOT DISTINCT FROM %s AND "單位" IS NOT DISTINCT FROM %s AND "分工關鍵字" IS NOT DISTINCT FROM %s',
                            (expanded, agency, unit, keyword)
                        )
                    else:
                        logger.warning(f"Skipping {agency} - {unit} due to empty or invalid LLM result.")
                        
                conn.commit()
                logger.info("Successfully updated all units.")
                
    except Exception as e:
        logger.error(f"Database operation failed: {e}")

if __name__ == "__main__":
    main()
