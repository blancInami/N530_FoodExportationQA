---
applyTo: "**"
---

# N530_FoodExportationQA — AI Agent 開發規範

## 🔴 安全與操作準則（重要）
- **在實作完成後，無論如何都不允許自行進行 `git commit`、`git push` 等相關動作。**
- 實作完成後僅能報告變更檔案與驗證結果，並詢問使用者等待確認指令。
- 所有的 `git commit` 與 `git push` 必須由使用者明確主動發出具體指令時才可執行，或由使用者自行在終端機操作。

---

## 專案簡介

食品輸銷問答 API，基於 FastAPI + PostgreSQL (pgvector) + 遠端 Embedding/LLM Server 構建的 RAG 系統。支援中英雙語問答，回應包含負責單位、雙語解答與參考來源。

---

## 技術棧

| 類別 | 套件 |
|------|------|
| Web Framework | `fastapi` 0.115, `uvicorn` |
| Async DB Driver | `psycopg[binary]>=3.1` |
| ORM / Query Builder | `sqlalchemy[asyncio]>=2.0`（Core only，禁止 ORM） |
| Vector Type | `pgvector>=0.3.6`（`pgvector.sqlalchemy.Vector`） |
| Schema Validation | `pydantic` v2, `pydantic-settings` |
| HTTP Client | `httpx` (async) |
| Token Count | `tiktoken` (cl100k_base) |
| Keyword Match | `ahocorasick-rs` |
| 繁簡轉換 | `opencc-python-reimplemented` |

---

## 資料夾結構

```
app/
├── config.py               # Pydantic Settings，DSN 格式：postgresql+psycopg://...
├── database.py             # create_async_engine + async_sessionmaker，init_engine()/close_engine()
├── dependencies.py         # get_db_session() → AsyncSession（FastAPI Depends）
├── logging_config.py       # 結構化 logging 初始化
├── models.py               # SQLAlchemy Core Table() 定義（10 張資料表，含官方正規詞彙與知識文獻三表）
├── routers/
│   ├── qa.py               # POST/GET /api/v1/qa/ask, POST /api/v1/qa/ingest, POST /api/v1/qa/breakdown
│   └── translate.py        # POST /api/v1/translation/re-translate
├── schemas/
│   ├── qa.py               # AskRequest, AskResponse, ReferenceSource, KnowledgeReferenceSource, IngestRequest/Response
│   ├── translate.py        # TranslateRequest, TranslateResponse
│   └── breakdown.py        # BreakdownItem(question_id, question_text)
├── services/
│   ├── embedding.py        # get_embedding(text) / get_embeddings_batch(texts)
│   ├── llm.py              # chat_completion(prompt, temperature, max_tokens)
│   ├── intent.py           # IntentClassifier 雙層意圖分類：Layer 1 Aho-Corasick, Layer 2 llm_fallback_classify()
│   │                       # _ALIAS_MAPPING_PATH → 從 app/resources/alias_mapping.json 載入別名擴展（tools/extract_aliases.py 離線產出）
│   │                       # _all_unit_rows 儲存全量列供 LLM 兜底分類（load() 期間轉存，無額外 DB 查詢）
│   ├── retrieval.py        # hybrid_retrieve(query, session, threshold, top_n) → RetrievalResult（Track A 問卷 + Track B 知識文獻）
│   ├── ingest.py           # ingest_questionnaire(pk, session) → dict
│   ├── translate.py        # translate_with_dictionary(text, session) → str
│   └── breakdown.py        # extract_questions(markdown) → list[dict]; convert_file_to_markdown()
├── utils/
│   ├── __init__.py         # sanitize_text_for_db(text), sliding_window_chunk(text, size, overlap)
│   │                       # normalize_for_matching(text)→str（清除標點空白，供 Aho-Corasick 一致性比對）
│   ├── dictionary.py       # DictionaryMatcher(Aho-Corasick), get_dictionary_matcher(session),
│   │                       # filter_by_text(*texts)→dict（Phase 1c）
│   │                       # filter_by_text_grounded(query, context)→dict（Phase 3，防短詞誤觸）
│   │                       # build_dictionary_xml(items)→str, load_dictionary(session)→dict（tools/ 全量載入用）
│   │                       # _TERM_MAPPING_PATH → 從 app/resources/term_mapping.json 載入短詞映射（tools/extract_stems.py 離線產出）
│   ├── connection_manager.py # monitor_disconnect(request), run_interruptible(coro, signal)（中斷與資源釋放機制）
│   └── opencc_converter.py # s2t(text) → str
└── resources/              # 離線工具產出的靜態資源（不入版控；執行對應工具後產生）
    ├── alias_mapping.json  # {標準分工關鍵字: [別名, ...]}，由 tools/extract_aliases.py 產出，供 intent.py 載入
    └── term_mapping.json   # {短詞: 官方術語}，由 tools/extract_stems.py 產出，供 dictionary.py 載入
```

### tools/ — 獨立離線工具（與主應用無相依）

```
tools/
├── requirements.txt             # PyPDF2, requests, openai>=1.0, httpx, psycopg[binary], markitdown>=0.1
├── ingest_agent.py              # CLI：文件 → markitdown（→ doc-to-json 降級）→ Gemma-4 LLM → JSON nodes → INSERT 知識文獻三表
│                                #   降級 A：markitdown 結果 < 50 字元時改用 doc-to-json API
│                                #   降級 B：LLM 萃取 0 節點時改用 doc-to-json 重試（僅 PDF）
├── batch_ingest_regulations.py  # CLI：批次掃描法規資料夾（REGULATION/GUIDELINE/QA 子資料夾），
│                                #   依子資料夾名稱自動決定 doc_type，逐一呼叫 ingest_agent.py
├── extract_stems.py             # CLI：官方正規詞彙 → 規則式剝後綴 + Gemma-4 → {短詞: 官方術語} JSON
│                                #   輸出：app/resources/term_mapping.json（供 dictionary.py DictionaryMatcher 載入）
├── extract_aliases.py           # CLI：單位對照表 → 分批 Gemma-4 → {標準分工關鍵字: [別名, ...]} JSON
│                                #   輸出：app/resources/alias_mapping.json（供 intent.py IntentClassifier 載入）
├── markdown_converter.py        # convert_to_markdown(file_path) → str（同步 markitdown 封裝）
├── extract_terms.py             # CLI：PDF 配對 → doc-to-json API → LLM 擷取 → 反向驗證 → CSV
├── pdf_reader.py                # read_pdf() via doc-to-json REST API（中英文統一流程）
├── db_extractor.py              # CLI：問卷題目檔 HTML → LLM 擷取 → 寫入官方正規詞彙
├── llm_client.py                # extract_terms_from_chunk() / extract_terms_from_html()（Gemma 4）
├── file_pairing.py              # discover_pdf_pairs()（遞迴走訪子資料夾）
├── chunking.py                  # 文本切塊 + 比例索引對齊
├── validator.py                 # 反向驗證（子字串比對防幻覺）
└── aggregator.py                # 詞彙頻率聚合 + CSV 匯出
```

> **tools/ 規則**：使用同步 `psycopg`（非 async）、`openai.OpenAI`（非 AsyncClient）。
> 不得引用 `app/` 任何模組。LLM Server：`http://10.166.57.22:40041/v1`，model=`gemma-4-26B-A4B-it-mtp`。

---

## 資料庫 Schema（完整）

> ⚠️ 禁止新增、修改或刪除任何現有欄位

```sql
-- 問卷主檔：問卷基本資訊
CREATE TABLE IF NOT EXISTS public."問卷主檔" (
    "主鍵" character varying(40),
    "輸出國家" character varying(40),
    "輸出品項" character varying(40),
    "問卷名稱" character varying(200),
    "發文日期" date,
    "發文文號" character varying(200),
    "主辦機關" character varying(200),
    "協辦機關" character varying(200),
    "是否刪除" integer,
    "刪除人員" character varying(50),
    "刪除日期" timestamp,
    "建立人員" character varying(50),
    "建立日期" timestamp,
    "修改人員" character varying(50),
    "修改日期" timestamp
);

-- 問卷題目檔：題目與回覆原文
CREATE TABLE IF NOT EXISTS public."問卷題目檔" (
    "主鍵" character varying(40),
    "問卷主檔主鍵" character varying(40),
    "上層題目主鍵" character varying(40),
    "題目序號" character varying(20),
    "題目" character varying NOT NULL,
    "回覆" character varying,
    "排序" integer NOT NULL,
    "發文日期" date,
    "是否刪除" integer,
    "刪除人員" character varying(50),
    "刪除日期" timestamp,
    "建立人員" character varying(50),
    "建立日期" timestamp NOT NULL,
    "修改人員" character varying(50),
    "修改日期" timestamp
);

-- 問卷題目切塊：向量化後的切塊資料（單向量欄位，chunk_source 標記來源）
CREATE TABLE IF NOT EXISTS public."問卷題目切塊" (
    "主鍵"           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    "問卷題目檔主鍵" character varying(40),
    "chunk_source"   character varying(50) NOT NULL,  -- 'question' | 'answer'
    "chunk_text"     text,
    "embedding"      vector(1024),
    "切塊索引"       integer NOT NULL,
    "詞元數量"       integer
);

-- 問卷附件檔：附件超連結
CREATE TABLE IF NOT EXISTS public."問卷附件檔" (
    "主鍵" character varying(40),
    "問卷主檔主鍵" character varying(40),
    "檔案名稱" character varying(200),
    "是否刪除" integer NOT NULL,
    "刪除人員" character varying(50),
    "刪除日期" timestamp,
    "建立人員" character varying(50),
    "建立日期" timestamp NOT NULL,
    "修改人員" character varying(50),
    "修改日期" timestamp,
    "檔案路徑" character varying(500)
);

-- 問卷原始檔：原始上傳檔案
CREATE TABLE IF NOT EXISTS public."問卷原始檔" (
    "主鍵" character varying(40),
    "問卷主檔主鍵" character varying(40),
    "檔案名稱" character varying(200),
    "是否刪除" integer NOT NULL,
    "刪除人員" character varying(50),
    "刪除日期" timestamp,
    "建立人員" character varying(50),
    "建立日期" timestamp NOT NULL,
    "修改人員" character varying(50),
    "修改日期" timestamp,
    "檔案路徑" character varying(500)
);

-- 單位對照表：分工關鍵字 → 機關/單位 對照
CREATE TABLE IF NOT EXISTS public."單位對照表" (
    "機關" character varying,
    "單位" character varying,
    "分工關鍵字" character varying,
    "擴充關鍵字" text[]
);

-- 官方正規詞彙：中英雙語術語字典（DictionaryMatcher Aho-Corasick 使用）
CREATE TABLE IF NOT EXISTS public."官方正規詞彙" (
    "中文字詞" character varying,
    "英文字詞" character varying
);

-- 知識文獻主檔：文獻基本資訊（由 tools/ingest_agent.py 離線寫入）
CREATE TABLE IF NOT EXISTS public."知識文獻主檔" (
    "主鍵"         character varying(40) PRIMARY KEY,
    "文獻名稱"     character varying(500) NOT NULL,
    "文獻類型"     character varying(20) NOT NULL CHECK ("文獻類型" IN ('REGULATION','GUIDELINE','QA')),
    "原始檔案路徑" character varying(500),
    "是否刪除"     integer DEFAULT 0,
    "建立日期"     timestamp NOT NULL DEFAULT now()
);

-- 知識文獻節點檔：文獻的邏輯節點（章節 / 條號 / QA 項）
CREATE TABLE IF NOT EXISTS public."知識文獻節點檔" (
    "主鍵"         character varying(40) PRIMARY KEY,
    "文獻主檔主鍵" character varying(40) NOT NULL,
    "節點標題路徑" character varying(500) NOT NULL,
    "節點內容"     character varying NOT NULL,
    "排序索引"     integer NOT NULL DEFAULT 1,
    "是否刪除"     integer DEFAULT 0,
    "建立日期"     timestamp NOT NULL DEFAULT now()
);

-- 知識文獻切塊檔：向量化後的切塊資料（由 tools/ingest_agent.py 離線寫入）
CREATE TABLE IF NOT EXISTS public."知識文獻切塊檔" (
    "主鍵"         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    "文獻節點檔主鍵" character varying(40) NOT NULL,
    "切塊內容"     character varying,
    "內容向量"     vector(1024),
    "切塊索引"     integer NOT NULL,
    "詞元數量"     integer
);
```

---

## Coding Conventions

### 1. 資料庫操作規則

- **嚴禁 Raw SQL 字串**（f-string 拼接 SQL、`conn.fetch()` 等）
- **嚴禁 ORM**（`DeclarativeBase`、`Mapped`、`session.add()` 等 ORM 模式）
- 所有查詢使用 **SQLAlchemy Core 表達式**：`select()` / `insert()` / `delete()` / `update()`
- 複雜查詢使用 `.cte()` 鏈式 CTE（不使用 Raw SQL WITH 語法）
- 向量欄位：`Column("題目向量", Vector(1024))`，寫入時直接傳 Python `list[float]`
- 條件篩選使用 `.in_(list)` 等方法（自動參數化，避免 SQL injection）
- PostgreSQL 保留字（如 `offset`、`limit`、`order`）不可作為 CTE 別名

### 2. 非同步規則

- 所有 DB 操作必須透過 `AsyncSession`（由 `get_db_session()` 依賴注入）
- `init_engine()` 在 FastAPI lifespan 啟動時呼叫；`close_engine()` 在關閉時呼叫
- HTTP 呼叫使用 `httpx.AsyncClient`，設定合適 timeout
- **中斷保護機制**：長耗時任務 (如向量檢索、LLM 呼叫) 需套用 `monitor_disconnect` (底層使用 `await request.receive()`) 與 `run_interruptible` 確保能即時捕捉斷線事件並拋出 `CancelledError`，中介軟體會攔截並回傳 `HTTP 499`

### 3. 向量相似度

- 使用 `.cosine_distance(vector)` 方法（自動生成 `<=>` SQL 運算子）
- 距離值越小代表越相似（`HAVING min_distance <= threshold`）
- 預設閾值 `0.75`，可透過 API 參數覆蓋

### 4. 文字處理

- 所有寫入 DB 的文字必須先經 `sanitize_text_for_db()` 清除控制字元
- 所有 LLM 輸出的中文必須經 `s2t()` 轉換為正體中文
- 英文輸入先翻譯為繁體中文後，再用中文文本做意圖分類與向量搜尋

### 5. Schema 規則

- Response model 欄位名稱使用中文（`負責單位`、`原始題目` 等）
- `AskResponse` 扁平化設計，其餘列表如下：
  - `負責單位` 為 `list[dict]`（機關與單位對照）
  - `參考來源` 為 `list[ReferenceSource]`
  - `知識參考來源` 為 `list[KnowledgeReferenceSource]`
- `ReferenceSource` 包含 5 個欄位：`問卷主檔主鍵`、`問卷名稱`、`輸出國家`、`輸出品項`、`題目序號`
- `KnowledgeReferenceSource` 包含 4 個欄位：`文獻主檔主鍵`、`文獻名稱`、`文獻類型`、`節點標題路徑`

---

## 主要業務邏輯（/qa/ask 管線）

```
Phase 1a：若輸入非中文 → LLM 翻譯成繁體中文（chinese_query）
Phase 1b：意圖分類（雙層，支援多重匹配）
           Layer 1 — Aho-Corasick 比對 chinese_query → list[負責機關/單位]（含原始、擴充關鍵字與別名）
           Layer 2 — LLM 兜底：Aho-Corasick 無命中或不足時，提供機關單位「聚合關鍵字描繪」予 LLM
                     LLM 採思維鏈 (CoT) 推導並輸出 XML 封裝之 JSON 列表。
Phase 1c：DictionaryMatcher.filter_by_text(chinese_query) → 術語特徵擴增
           命中前 5 筆術語附加至查詢（enriched_query）
Phase 2 ：Embed enriched_query → 1024-dim 向量；雙軌並行 SQLAlchemy Core CTE 鏈：

   Track A 問卷庫（出自問卷題目切塊）：
   chunk_hits CTE      (cosine_distance embedding 單向量, LIMIT top_k)
   aggregated CTE      (GROUP BY 問卷題目檔主鍵+chunk_source+切塊索引, MIN distance, HAVING <= threshold, LIMIT top_n)
   adj        CTE      (VALUES -1/0/1 展開相鄰切塊)
   expanded   CTE      (CROSS JOIN aggregated × adj，攜帶 chunk_source)
   expanded_chunks CTE (JOIN 問卷題目切塊，限定同 chunk_source + neighbor_idx)
   final_data CTE      (JOIN 問卷題目檔 is_deleted=0，取完整題目/回覆原文 + 題目序號)
   → JOIN 問卷主檔 取 問卷名稱/輸出國家/輸出品項
   col.in_(master_pks) → JOIN 問卷附件檔 取 檔案路徑（全參數化）
   raw_context 組裝時以問卷題目檔主鍵去重（多切塊→同題只取一次完整 Q/A 原文）

   Track B 知識文獻庫（出自知識文獻切塊檔）：
   knowledge_hits CTE          (cosine_distance 內容向量, LIMIT top_k)
   knowledge_agg CTE           (GROUP BY, MIN distance, HAVING <= threshold, LIMIT top_n)
   knowledge_adj CTE           (VALUES -1/0/1)
   knowledge_expanded CTE      (CROSS JOIN knowledge_agg × knowledge_adj)
   knowledge_expanded_chunks CTE (JOIN 知識文獻切塊檔 取切塊內容)
   knowledge_final CTE         (JOIN 知識文獻節點檔 取節點標題路徑) → JOIN 知識文獻主檔 取文獻名稱/文獻類型
   Track B 結果產生 knowledge_context（依 REGULATION/GUIDELINE/QA 類型格式化 XML 區塊）

   Python 層全局合並：依 `settings.retrieval_merge_mode` 決定策略
     - `independent`（預設）：兩軌各自獨立保留 top_n，知識文獻不被問卷擠出
     - `compete`：兩軌混排後共用 top_n 配額（由 RETRIEVAL_MERGE_MODE 環境變數控制）
Phase 3 ：DictionaryMatcher.filter_by_text(chinese_query, raw_context)（複用 Phase 1c singleton）
          → 從「官方正規詞彙」6000+ 筆中篩出命中術語（~10-50 筆）
          → build_dictionary_xml() 組 XML 注入 LLM prompt
          → 若有知識文獻命中，附加 <Knowledge_References> 區塊
          → chat_completion() → 解析 "---" 分隔線 → (英文回覆, 中文回覆)
Phase 4 ：s2t(中文回覆) 確保正體中文；回傳 9 欄位扁平化 AskResponse
         （負責單位/原始題目/中文譯文/中文回覆/英文回覆/附件超連結/參考來源/知識參考來源）
```

## tools/ingest_agent.py 離線知識匯入管線

```
1. CLI 參數：--input（必填）、--doc-name（必填）、--doc-type（選填覆寫）、--is-markdown、--verbose
2. markitdown 轉換 → macro_chunk_markdown()（H1/H2 切分 + 硬上限 3000 字）
   降級 A：markitdown 結果 < 50 字元（MARKITDOWN_MIN_LENGTH）且為 PDF → 改用 doc-to-json API
3. 每個 macro_chunk → Gemma-4 LLM → JSON nodes（節點標題路徑 + 節點內容 + 文獻類型）
   降級 B：LLM 萃取 0 節點 且尚未使用過 doc-to-json → 改用 doc-to-json 重試（僅 PDF）
4. 多數決 doc_type 寫入知識文獻主檔
5. 每個節點 → sliding_window_chunk → get_embeddings_batch → INSERT 知識文獻節點檔 + 知識文獻切塊檔
```

## tools/batch_ingest_regulations.py 批次匯入管線

```
1. 掃描根資料夾下三個子資料夾（REGULATION / GUIDELINE / QA）
2. 依子資料夾名稱自動決定 doc_type
3. 以檔名（不含副檔名）作為 doc_name，逐一呼叫 ingest_agent.py
4. 最終輸出成功/失敗統計摘要；任一失敗以 exit code 1 結束
```

## /qa/breakdown 管線

```
1. 驗證副檔名（pdf / doc / docx / xlsx / xls）
2. 寫入暫存檔（tempfile.NamedTemporaryFile）
3. asyncio.to_thread(convert_file_to_markdown, path, ext) — MarkItDown（CPU-bound）
4. 滑動視窗切塊 → asyncio.gather + Semaphore → LLM 萃取題號（extract_questions）
5. 清除暫存檔；回傳 list[BreakdownItem]
```

---

## 禁止事項

| 禁止行為 | 原因 |
|----------|------|
| 使用 ORM（`DeclarativeBase`、`session.add()` 等） | 架構規範 |
| 使用 f-string 或字串拼接組成 SQL | SQL Injection 風險 |
| 修改 `問卷主檔` 等現有資料表欄位 | 系統整合約束 |
| 在 CTE 別名中使用 `offset`、`limit`、`order` | PostgreSQL 保留字 |
| 對 LLM 輸出中文不做 OpenCC 轉換 | 可能出現簡體字 |
| 暴露 `top_k` 參數給使用者 | 由系統自動計算 `max(TOP_K, top_n+25)` |
| 跳過 `sanitize_text_for_db` 直接寫入 | 控制字元可能導致 DB 錯誤 |

---

## 外部服務介面

### Embedding Server（OpenAI compatible）

```
POST http://{EMBEDDING_HOST}:{EMBEDDING_PORT}/v1/embeddings
Body: { "model": "intfloat/multilingual-e5-large-instruct", "input": "..." }
Returns: data[0].embedding (list[float], len=1024)
```

### LLM Server（OpenAI compatible）

```
POST http://{LLM_HOST}:{LLM_PORT}/v1/chat/completions
Body: { "model": "gpt-oss-20b", "messages": [...], "temperature": 0.3 }
Returns: choices[0].message.content (str)
```

---

## 環境變數（完整列表）

```env
EMBEDDING_HOST=10.166.57.22
EMBEDDING_PORT=40003
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
EMBEDDING_DIM=1024

LLM_HOST=10.166.57.22
LLM_PORT=40036
LLM_MODEL=gpt-oss-20b

PG_HOST=127.0.0.1
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=postgres
PG_DB=fes

SIMILARITY_THRESHOLD=0.75
TOP_K=30
TOP_N=5

# RETRIEVAL_MERGE_MODE: independent（預設）— 兩軌各自保留 top_n，不競爭名額
#                       compete       — 兩軌混排後共用 top_n 配額（舊行為）
RETRIEVAL_MERGE_MODE=independent

CHUNK_SIZE=500
CHUNK_OVERLAP=50

# doc-to-json 降級 API（ingest_agent.py 備援用，主應用不使用）
DOC2JSON_API_URL=http://10.166.57.22:43002/api/doc-to-json/run
```