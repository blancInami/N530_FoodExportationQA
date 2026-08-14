# N530_FoodExportationQA — 食品輸銷問答 API

基於 RAG（Retrieval-Augmented Generation）架構的食品輸銷知識雙語問答系統，整合問卷、法規、作業指引與問答文獻。

---

## 系統架構

```
使用者提問
    ├─ Phase 1a [非中文] → LLM 翻譯成繁體中文
    │
    ├─ Phase 1b 意圖分類（雙層，支援多重回傳）
    │       ├─ Layer 1：Aho-Corasick 關鍵字比對（含分工關鍵字 + 擴充關鍵字 + alias_mapping.json 別名）
    │       └─ Layer 2 兜底：LLM 分類（Aho-Corasick 無命中或不足時觸發，採聚合關鍵字 + CoT 推導）
    │
    ├─ Phase 1c DictionaryMatcher 特徵擴增
    │       └─ 從「官方正規詞彙」命中術語，附加至查詢字串（enriched_query）
    │
    ├─ Embedding Server (multilingual-e5-large-instruct)
    │       └─ enriched_query 向量化
    │
    ├─ Phase 2 向量檢索（雙軌並行）
    │   ├─ Track A 問卷問答庫
    │   │       ├─ 單一 embedding 向量搜尋，以 chunk_source 區分 question / answer
    │   │       ├─ CROSS JOIN 同來源相鄰切塊展開 (±1 切塊)
    │   │       └─ JOIN 問卷主檔 / 問卷附件檔 / 問卷題目檔
    │   └─ Track B 知識文獻庫
    │           ├─ cosine_distance 向量搜尋 (內容向量)
    │           ├─ CROSS JOIN 展開相鄰切塊
    │           └─ JOIN 知識文獻節點檔 / 知識文獻主檔（取節點標題路徑、文獻名稱、文獻類型）
    │
    ├─ Phase 3 LLM 雙語生成
    │       ├─ DictionaryMatcher 動態篩選術語 → XML 注入 Prompt
    │       ├─ <Knowledge_References> 知識文獻內容區塊（若有命中）
    │       └─ XML-structured Prompt → 雙語回覆 (英文 + 繁體中文)
    │
    └─ Phase 4 回傳組裝 → OpenCC s2t → AskResponse → 非同步寫入查詢紀錄


系統模組分層（app/services/）
  ├─ breakdown/              ─ 問卷解析引擎家族（v1 / v2 / vlm / langgraph / preprocessing / dlq）
  ├─ core/                   ─ 核心基礎服務（llm / embedding / translate）
  └─ qa/                     ─ 問答檢索與入庫（retrieval / intent / ingest / query_log）

獨立離線工具（tools/）
  ├─ evaluation/             ─ 問卷解析精準度評估工具（全引擎支援）
  │    ├─ evaluate_breakdown.py
  │    ├─ evaluate_breakdown_v2.py
  │    ├─ evaluate_breakdown_vlm.py
  │    └─ evaluate_breakdown_langgraph.py
  ├─ term_extraction/        ─ 雙語專有名詞、別名與詞幹離線擷取工具鏈
  │    ├─ extract_terms.py
  │    ├─ extract_stems.py
  │    ├─ extract_aliases.py
  │    └─ db_extractor.py
  ├─ ingestion/              ─ 法規與知識文獻離線批次入庫工具
  │    ├─ ingest_agent.py
  │    └─ batch_ingest_regulations.py
  └─ data/                   ─ 離線萃取之 CSV 專有名詞辭典檔案
```

---

## 快速啟動

### 1. 前置需求

- Python 3.11+
- PostgreSQL with `pgvector` extension installed，或 SQL Server 2025（原生 `VECTOR` 型別）
- Embedding Server & LLM Server running (see `.env`)

### 2. 建立虛擬環境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

### 4. 設定環境變數

複製 `.env` 並依實際環境調整：

```bash
cp .env .env.local
```

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `EMBEDDING_HOST` | `10.166.57.22` | Embedding Server IP |
| `EMBEDDING_PORT` | `40003` | Embedding Server Port |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large-instruct` | 模型名稱 |
| `LLM_HOST` | `10.166.57.22` | LLM Server IP |
| `LLM_PORT` | `40036` | LLM Server Port |
| `LLM_MODEL` | `gpt-oss-20b` | 模型名稱 |
| `DB_TYPE` | `postgres` | 資料庫後端：`postgres` 或 `mssql` |
| `DB_SCHEMA` | `public` | Schema：PostgreSQL 預設 `public`；SQL Server 預設 `dbo` |
| `PG_HOST` | `127.0.0.1` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_USER` | `postgres` | 使用者 |
| `PG_PASSWORD` | `postgres` | 密碼 |
| `PG_DB` | `fes` | 資料庫名稱 |
| `MSSQL_HOST` | `127.0.0.1` | SQL Server host（`DB_TYPE=mssql` 時使用） |
| `MSSQL_PORT` | `1433` | SQL Server port |
| `MSSQL_USER` | `sa` | SQL Server 使用者 |
| `MSSQL_PASSWORD` | 空字串 | SQL Server 密碼 |
| `MSSQL_DB` | `fes` | SQL Server 資料庫名稱 |
| `MSSQL_DRIVER` | `ODBC Driver 18 for SQL Server` | SQL Server ODBC 驅動程式 |
| `MSSQL_TRUST_CERT` | `false` | 是否信任 SQL Server 憑證 |
| `SIMILARITY_THRESHOLD` | `0.75` | 向量相似度篩選閾值 (0~1) |
| `TOP_K` | `30` | 初步向量搜尋候選數量 |
| `TOP_N` | `5` | 最終聚合取回筆數 |
| `INTENT_TOP_N` | `1` | 意圖分類回傳的負責單位數量 |
| `CHUNK_SIZE` | `500` | 切塊字元大小 |
| `CHUNK_OVERLAP` | `50` | 切塊重疊字元數 |
| `BREAKDOWN_MAX_CONCURRENCY` | `5` | `/qa/breakdown` 全 LLM fallback 的最大並行請求數 |
| `BREAKDOWN_LLM_VERIFY` | `false` | 僅在 breakdown 的 source_key 有遺漏或重複時，以 LLM 做唯讀二次驗證；不允許改寫題號 |
| `BREAKDOWN_NFKC_NORMALIZE` | `true` | Breakdown 可視文本採用 Unicode NFKC 正規化；不改寫 Markdown link target、HTML/XML 標籤與 fenced code |
| `BREAKDOWN_TABLE_LINEARIZATION_MODE` | `off` | 表格線性化預留開關。目前主解析器仍使用原始 Markdown pipe table，設定非 `off` 時僅記錄警告 |
| `BREAKDOWN_DLQ_ENABLED` | `false` | 是否啟用 breakdown JSONL dead-letter queue |
| `BREAKDOWN_DLQ_PATH` | `logs/breakdown-dlq.jsonl` | JSONL dead-letter queue 的輸出路徑 |
| `BREAKDOWN_DLQ_INCLUDE_RAW_PAYLOAD` | `false` | 是否在 DLQ 保存截斷後原始內容；預設僅保存 SHA-256 摘要 |
| `RETRIEVAL_MERGE_MODE` | `independent` | 雙軌合併策略：`independent`（各軌獨立取 top_n）/ `compete`（兩軌共用 top_n 配額） |

### 5. 啟動服務

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

互動式 API 文件：http://localhost:8000/docs

---

## API 規格

### `POST /api/v1/qa/ask` / `GET /api/v1/qa/ask`

單題問答。輸入問題，系統自動進行意圖分類、向量檢索、雙語生成。

**Request Body (POST)**

```json
{
  "question": "原料之加工或生產是否依衛生方式執行？",
  "similarity_threshold": 0.75,
  "top_n": 5,
  "intent_top_n": 1
}
```

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `similarity_threshold` | float (0~1) | ❌ | 覆蓋預設相似度閾值 |
| `top_n` | int (1~100) | ❌ | 覆蓋預設最終回傳筆數；`top_k` 自動計算為 `max(TOP_K, top_n+25)` |
| `intent_top_n` | int (1~10) | ❌ | 覆蓋預設意圖分類回傳單位數 |

**Response**

```json
{
  "負責單位": [{"機關": "農業部", "單位": "動植物防疫檢疫署"}],
  "原始題目": "原料之加工或生產是否依衛生方式執行？",
  "中文譯文": "",
  "中文回覆": "根據相關法規，原料之屠宰、調味、去除內臟與分切，均需依...",
  "英文回覆": "According to relevant regulations, the processing of raw materials...",
  "附件超連結": ["/files/attach/abc123.pdf"],
  "參考來源": [
    {
      "問卷主檔主鍵": "Q20240101001",
      "問卷名稱": "美國豬肉輸出問卷",
      "輸出國家": "美國",
      "輸出品項": "豬肉",
      "題目序號": "3.1"
    }
  ],
  "知識參考來源": [
    {
      "文獻主檔主鍵": "K20240001",
      "文獻名稱": "食品安全衛生管理法",
      "文獻類型": "REGULATION",
      "節點標題路徑": "食品安全衛生管理法 > 第7條"
    }
  ]
}
```

> 若輸入為英文，`中文譯文` 欄位將填入 LLM 翻譯結果；若無問卷及知識文獻命中資料則回傳「無法作答」訊息。

---

### `POST /api/v1/qa/ingest`

問卷切塊向量化。接收問卷主檔主鍵**清單**，依序對每筆問卷執行滑動視窗切塊並寫入向量資料庫。

**Request Body**

```json
{
  "問卷主檔主鍵清單": ["Q20240101001", "Q20240101002"]
}
```

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `問卷主檔主鍵清單` | list[string] | ✅ | 一或多筆問卷主檔主鍵 |

**Response**

```json
{
  "total_questionnaires": 2,
  "total_items_processed": 42,
  "total_chunks_written": 154,
  "results": [
    {
      "問卷主檔主鍵": "Q20240101001",
      "status": "success",
      "items_processed": 25,
      "total_chunks_written": 87,
      "message": ""
    },
    {
      "問卷主檔主鍵": "Q20240101002",
      "status": "success",
      "items_processed": 17,
      "total_chunks_written": 67,
      "message": ""
    }
  ]
}
```

**說明**
- 清單中每筆問卷**依序**處理，單筆失敗不中斷，`status` 欄位顯示 `"error"` 並於 `message` 記錄原因
- 切塊演算法：滑動視窗，預設 `chunk_size=500`、`overlap=50` 字元
- Token 計數：tiktoken `cl100k_base`
- 若舊切塊存在，先 DELETE 再重新寫入

---

### `POST /api/v1/translate`

官方正規詞彙翻譯。以強制詞彙字典確保專有名詞正確翻譯。

**Request Body**

```json
{
  "text": "請提供衛生福利部食品藥物管理署核准的健康證明書。"
}
```

**Response**

```json
{
  "original": "請提供衛生福利部食品藥物管理署核准的健康證明書。",
  "translation": "Please provide a Health Certificate approved by the Taiwan Food and Drug Administration (TFDA)."
}
```

> **端點路徑**：`POST /api/v1/translation/re-translate`

---

### `POST /api/v1/qa/breakdown`

問卷檔案題號萃取。上傳問卷原始檔（PDF / Word / Excel），自動解析並萃取所有英文題號與題目。

**Request**：`multipart/form-data`，欄位名稱 `file`，支援格式：`pdf`, `doc`, `docx`, `xlsx`, `xls`

**Response**

```json
[
  {
    "question_id": "2.2.1",
    "question_text": "Are raw materials processed in a hygienic manner?"
  },
  {
    "question_id": "1.2.4.a",
    "question_text": "Provide details on certification equivalent to Directive 96/93/EC."
  }
]
```

---

#### 核心解析引擎與架構模式（可由 `.env` 中的 `BREAKDOWN_ENGINE` 切換）

系統支援四種解析引擎，適應不同問卷複雜度與視覺排版要求：

1. **`langgraph`（【推薦】擬人化循序閱讀與主動回讀狀態機）**：
   - **實作模組**：[`app/services/breakdown_langgraph.py`](file:///C:/Users/6747/Desktop/Projekt/2026/N530/N530_FoodExportationQA/app/services/breakdown_langgraph.py)
   - **核心機制**：
     - **轉檔預處理**：透過 `app/lo/file_utils.py`（LibreOffice + Poppler）將 Word / PDF 展開為高解析度逐頁 JPEG。
     - **Node 1 (TOC 導航)**：分析前置頁建立全局大綱樹（`toc_structure`）。
     - **Node 2 (逐頁研讀)**：由上而下閱讀，優先縫合上一頁遺留的 `pending_context`，並萃取實質問卷題目。
     - **Node 3 (條件路由)**：若偵測到題號斷裂或跨頁斷層，觸發回讀請求。
     - **Node 4 (記憶回讀審查)**：從 Checkpoint 歷史調取前頁原始影像進行雙頁對照與修正。
     - **Node 5 (全局統整)**：全篇完成後由 LLM 執行跨頁去重、題號標準化與偽題目二次過濾。

2. **`vlm`（視覺多模態並行解析）**：
   - **實作模組**：[`app/services/breakdown_vlm.py`](file:///C:/Users/6747/Desktop/Projekt/2026/N530/N530_FoodExportationQA/app/services/breakdown_vlm.py)
   - **核心機制**：
     - **Pass 1 骨架掃描**：快速建立大綱樹並為每頁計算 `Active Ancestor Path`。
     - **Pass 2 並行推論**：注入主章節路徑與嚴格題目性過濾規則，多頁並行送入 Vision 模型。
     - **Pass 3 一致性校驗**：跨頁文字無縫拼接與全局雜訊清洗。

3. **`v2`（純文字 LLM-First 全域骨架方案 A）**：
   - **實作模組**：[`app/services/breakdown_v2.py`](file:///C:/Users/6747/Desktop/Projekt/2026/N530/N530_FoodExportationQA/app/services/breakdown_v2.py)
   - **核心機制**：
     - 將檔案轉為純文字 Markdown。
     - Pass 1 建立大綱樹 $\rightarrow$ 切塊綁定祖先路徑 $\rightarrow$ Pass 2 並行 LLM 萃取 $\rightarrow$ Pass 3 全局審核。

4. **`v1`（舊版 Regex-Heavy 狀態機）**：
   - **實作模組**：[`app/services/breakdown.py`](file:///C:/Users/6747/Desktop/Projekt/2026/N530/N530_FoodExportationQA/app/services/breakdown.py)
   - 採 deterministic 表格快速路徑 + 正規表達式狀態機 + LLM 輔助填入題目文字。

---

#### 問卷解析評估工具（`tools/`）

提供統一與專屬評估工具，對照黃金標準答案（`tests/data/*_解析結果.json`）評估 Precision、Recall 與 F1：

```bash
# 1. 統一評估工具（支援所有模式：--mode langgraph | vlm | v2 | hybrid | structured）
venv\Scripts\python.exe tools/evaluate_breakdown.py --input "tests/data/1_輸日禽畜肉品.docx" --expected "tests/data/1_輸日禽畜肉品_解析結果.json" --mode langgraph

# 2. 專屬 LangGraph 評估工具
venv\Scripts\python.exe tools/evaluate_breakdown_langgraph.py --input "tests/data/3_澳洲水產品.md" --expected "tests/data/3_澳洲水產品_解析結果.json"

# 3. 專屬 V2 評估工具
venv\Scripts\python.exe tools/evaluate_breakdown_v2.py --all
```

# 執行 parser + LLM + 來源題號校正的完整評估
python tools/evaluate_breakdown.py --input "問卷.docx" --expected "人工解答.json" --mode hybrid --min-recall 0.95
```

當題號 recall 低於 `--min-recall` 時，工具會以 exit code `1` 結束，可作為回歸測試或 CI 品質閘門。
報告也會提供 `toc_lines_removed` 與 `toc_detection_strategy`，用以確認目錄清理是否生效。

---

```
N530_FoodExportationQA/
├── .env                        # 環境變數（不入版控）
├── requirements.txt
├── main.py                     # FastAPI 應用程式入口（含 lifespan、CORS、請求日誌）
└── app/
    ├── config.py               # Pydantic Settings；依 DB_TYPE 產生 PostgreSQL 或 SQL Server DSN
    ├── database.py             # SQLAlchemy async engine + sessionmaker，init_engine()/close_engine()
    ├── db_dialect.py           # 雙後端型別與向量距離抽象（MssqlVector、cosine_distance_expr）
    ├── dependencies.py         # get_db_session() → AsyncSession（FastAPI Depends）
    ├── logging_config.py       # 結構化 logging 初始化
    ├── models.py               # SQLAlchemy Core Table() 定義（11 張資料表，含查詢紀錄與知識文獻三表）
    ├── routers/
    │   ├── qa.py               # /api/v1/qa/ask, /api/v1/qa/ingest, /api/v1/qa/breakdown
    │   └── translate.py        # /api/v1/translation/re-translate
    ├── schemas/
    │   ├── qa.py               # AskRequest / AskResponse / ReferenceSource / KnowledgeReferenceSource / IngestRequest
    │   ├── translate.py        # TranslateRequest / TranslateResponse
    │   └── breakdown.py        # BreakdownItem
    ├── services/
    │   ├── embedding.py        # 呼叫遠端 Embedding Server
    │   ├── llm.py              # 呼叫遠端 LLM Server
    │   ├── intent.py           # IntentClassifier 雙層意圖分類：
    │   │                       #   Layer 1 Aho-Corasick（分工關鍵字 + 擴充關鍵字 + alias_mapping.json 別名）
    │   │                       #   Layer 2 llm_fallback_classify()（LLM 全量列兜底，命中數不足時觸發）
    │   ├── retrieval.py        # SQLAlchemy Core 雙軌 CTE 鏈式向量檢索（問卷 + 知識文獻）
    │   ├── ingest.py           # 問卷滑動視窗切塊 + tiktoken + SQLAlchemy insert
    │   ├── query_log.py        # /qa/ask 完整管線內容非同步寫入查詢紀錄
    │   ├── translate.py        # 官方詞彙翻譯（DictionaryMatcher 篩選）
    │   ├── breakdown.py        # 問卷檔案 → parser-owned candidates → LLM 文字萃取
    │   ├── breakdown_preprocessing.py # NFKC 正規化與 Markdown 表格線性化工具
    │   └── breakdown_dlq.py    # 可選 JSONL dead-letter queue writer
    ├── utils/
    │   ├── __init__.py         # sanitize_text_for_db, sliding_window_chunk, normalize_for_matching
    │   ├── dictionary.py       # DictionaryMatcher（Aho-Corasick）+ filter_by_text() + filter_by_text_grounded()
    │   │                       # + build_dictionary_xml()；從 term_mapping.json 載入短詞映射
    │   └── opencc_converter.py # OpenCC 簡繁轉換 (s2t)
    └── resources/              # 離線工具產出的靜態資源（不入版控；執行對應工具後產生）
        ├── alias_mapping.json  # {標準分工關鍵字: [別名, ...]}，由 tools/extract_aliases.py 產出
        └── term_mapping.json   # {短詞: 官方術語}，由 tools/extract_stems.py 產出

tools/                          # 離線工具（與主應用無相依，獨立安裝依賴）
    ├── requirements.txt        # 獨立依賴（PyPDF2, requests, openai, psycopg[binary], markitdown）
    ├── ingest_agent.py         # CLI 入口：文件 → markitdown（→ doc-to-json 降級）→ Gemma-4 LLM → JSON nodes → INSERT 知識文獻三表
    ├── batch_ingest_regulations.py # CLI 入口：批次掃描法規/REGULATION|GUIDELINE|QA 子資料夾，逐一呼叫 ingest_agent
    ├── extract_stems.py        # CLI 入口：官方正規詞彙 → 規則式剝後綴 + Gemma-4 → term_mapping.json
    ├── extract_aliases.py      # CLI 入口：單位對照表 → 分批 Gemma-4 → alias_mapping.json
    ├── expand_keywords.py      # CLI 入口：分工關鍵字 → LLM 擴充 → 回寫單位對照表.擴充關鍵字
    ├── db_utils.py             # 雙後端同步連線、Schema、占位符與陣列編碼抽象
    ├── markdown_converter.py   # convert_to_markdown()（同步 markitdown 封裝）
    ├── extract_terms.py        # CLI 入口：PDF 配對 → 擷取 → 匯出 CSV
    │                           #   --workers N       並行處理數（預設 1）
    │                           #   --no-structured   停用結構化解析（圖片型 PDF 用純 OCR）
    ├── pdf_reader.py           # doc-to-json REST API 讀取 PDF
    │                           #   read_pdf()            → 平文字字串（向下相容）
    │                           #   read_pdf_structured() → list[Block]（type/text/page/order）
    ├── db_extractor.py         # CLI 入口：問卷題目檔 HTML → 擷取 → 寫入官方正規詞彙
    ├── llm_client.py           # Gemma 4 LLM 客戶端 + JSON 解析容錯
    ├── file_pairing.py         # PDF 資料夾掃描 + 中英文配對（含遞迴走訪）
    ├── chunking.py             # 文本切塊 + 比例索引對齊
    │                           #   chunk_text()           字元滑動視窗（降級備用）
    │                           #   chunk_by_structure()   結構化語意切塊（tiktoken token budget，title 邊界，table 獨立）
    ├── validator.py            # 反向驗證（防幻覺：子字串比對）
    └── aggregator.py           # 詞彙頻率聚合 + CSV 匯出
                                #   英文字詞去重前統一轉小寫；CSV 欄位：中文字詞、英文字詞、出現頻率、負責單位、來源檔案
```

---

## 資料庫 Schema 摘要

| 資料表 | 用途 |
|--------|------|
| `問卷主檔` | 問卷基本資訊（國家、品項、名稱、機關） |
| `問卷題目檔` | 題目與回覆原文（HTML 格式，含中英文） |
| `問卷題目切塊` | 向量化切塊（`chunk_source` 區分 `question` / `answer`；單一 `embedding` vector(1024)） |
| `問卷附件檔` | 問卷相關附件檔案路徑 |
| `問卷原始檔` | 問卷原始上傳檔案 |
| `單位對照表` | 關鍵字 → 機關/單位 對照（Layer 1 Aho-Corasick 意圖分類；Layer 2 LLM 兜底全量列，含擴充關鍵字） |
| `官方正規詞彙` | 中文字詞 → 英文字詞（供 DictionaryMatcher Aho-Corasick 篩選） |
| `知識文獻主檔` | 文獻基本資訊（文獻名稱、類型 REGULATION/GUIDELINE/QA） |
| `知識文獻節點檔` | 文獻邏輯節點（節點標題路徑、節點內容） |
| `知識文獻切塊檔` | 知識文獻向量切塊（`內容向量` vector(1024)，由 tools/ingest_agent.py 離線管理） |
| `查詢紀錄` | 每次 /ask 呼叫的完整管線日誌 |

---

## 核心設計原則

1. **嚴禁 ORM**：所有資料庫查詢均使用 **SQLAlchemy Core 表達式**（`select()` / `insert()` / `delete()`），複雜查詢以 `.cte()` 鏈式組合
2. **Async 優先**：所有 DB 操作透過 `AsyncSession`；PostgreSQL 使用 psycopg3，SQL Server 使用 aioodbc；HTTP 呼叫使用 `httpx.AsyncClient`
3. **雙後端支援**：以 `DB_TYPE=postgres|mssql` 切換 PostgreSQL + pgvector 或 SQL Server 2025 原生 `VECTOR`；方言差異集中於 `app/db_dialect.py` 與 `tools/db_utils.py`
4. **向量格式**：應用層一律傳遞 Python `list[float]`；方言抽象層負責對應後端的綁定格式
5. **OpenCC 保證**：所有 LLM 輸出的中文均經 `opencc s2t` 確保正體中文
6. **不修改現有 DB 結構**：欄位定義依照既有 Schema，不新增或改動欄位
7. **全參數化查詢**：條件篩選使用 `.in_(list)` 等 SQLAlchemy 方法，自動綁定參數，杜絕 SQL Injection
8. **DictionaryMatcher 動態篩選**：`官方正規詞彙` 6000+ 筆字典透過 Aho-Corasick 自動機，掃描輸入文本後僅注入命中術語（~10-50 筆），避免塞爆 LLM context window
9. **優雅的斷線中斷機制**：使用 `monitor_disconnect` (依賴 `await request.receive()`) 監聽 ASGI 斷線事件，結合 `run_interruptible` 封裝長耗時任務，確保客戶端斷線時能立刻拋出 `asyncio.CancelledError`，釋放系統資源並阻止無效的資料庫寫入（回傳 499 狀態碼）
