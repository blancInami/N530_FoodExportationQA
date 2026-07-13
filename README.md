# N530_FoodExportationQA — 食品輸銷問答 API

基於 RAG（Retrieval-Augmented Generation）架構的食品輸銷法規雙語問答系統。

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
    ├─ Phase 2 PostgreSQL + pgvector（雙軌並行）
    │   ├─ Track A 問卷問答庫
    │   │       ├─ UNION ALL 雙路徑向量搜尋 (題目向量 + 回覆向量)
    │   │       ├─ CROSS JOIN 上下文切塊展開 (±1 切塊)
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
    └─ Phase 4 回傳組裝 → OpenCC s2t → 9 欄位 AskResponse


獨立離線工具（tools/）
    ├─ ingest_agent.py         ─ 文件 → markitdown（→ doc-to-json 降級）→ Gemma-4 LLM → INSERT 知識文獻三表
    ├─ batch_ingest_regulations.py ─ 批次掃描 法規/REGULATION|GUIDELINE|QA 子資料夾 → 逐一呼叫 ingest_agent
    ├─ extract_terms.py        ─ PDF 配對 → doc-to-json API（結構化/平文字雙模式）→ 語意切塊（chunk_by_structure）→ 平行 LLM 擷取 → 反向驗證 → CSV（含來源檔案）
    ├─ db_extractor.py         ─ 問卷題目檔 (HTML) → LLM 擷取 → 寫入官方正規詞彙
    ├─ extract_stems.py        ─ 官方正規詞彙 → 短詞映射 → app/resources/term_mapping.json（供 DictionaryMatcher 線上載入）
    └─ extract_aliases.py      ─ 單位對照表 → 機關別名映射 → app/resources/alias_mapping.json（供 intent.py Aho-Corasick 擴展）
```

---

## 快速啟動

### 1. 前置需求

- Python 3.11+
- PostgreSQL with `pgvector` extension installed
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
| `PG_HOST` | `127.0.0.1` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_USER` | `postgres` | 使用者 |
| `PG_PASSWORD` | `postgres` | 密碼 |
| `PG_DB` | `fes` | 資料庫名稱 |
| `SIMILARITY_THRESHOLD` | `0.75` | 向量相似度篩選閾值 (0~1) |
| `TOP_K` | `30` | 初步向量搜尋候選數量 |
| `TOP_N` | `5` | 最終聚合取回筆數 |
| `CHUNK_SIZE` | `500` | 切塊字元大小 |
| `CHUNK_OVERLAP` | `50` | 切塊重疊字元數 |
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

**說明**
- 多語系問卷自動捨棄非英文字元，僅保留英文題目
- 子題號自動合併為點分十進位格式（如 `2.2` 下的 `(1)` → `2.2.1`）
- 長文件自動切塊並行推論，部分失敗不中斷整體

---

```
N530_FoodExportationQA/
├── .env                        # 環境變數（不入版控）
├── requirements.txt
├── main.py                     # FastAPI 應用程式入口（含 lifespan、CORS、請求日誌）
└── app/
    ├── config.py               # Pydantic Settings，DSN 格式：postgresql+psycopg://...
    ├── database.py             # SQLAlchemy async engine + sessionmaker，init_engine()/close_engine()
    ├── dependencies.py         # get_db_session() → AsyncSession（FastAPI Depends）
    ├── logging_config.py       # 結構化 logging 初始化
    ├── models.py               # SQLAlchemy Core Table() 定義（10 張資料表，含知識文獻三表）
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
    │   │                       #   Layer 1 Aho-Corasick（分工關鍵字 + alias_mapping.json 別名）
    │   │                       #   Layer 2 llm_fallback_classify()（LLM 全量列兜底，僅 Aho-Corasick 無命中時觸發）
    │   ├── retrieval.py        # SQLAlchemy Core 雙軌 CTE 鏈式向量檢索（問卷 + 知識文獻）
    │   ├── ingest.py           # 問卷滑動視窗切塊 + tiktoken + SQLAlchemy insert
    │   ├── translate.py        # 官方詞彙翻譯（DictionaryMatcher 篩選）
    │   └── breakdown.py        # 問卷檔案 → Markdown → LLM 萃取題號
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
| `問卷題目切塊` | 向量化切塊（`題目向量` / `回覆向量` vector(1024)） |
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
2. **Async 優先**：所有 DB 操作透過 `AsyncSession`（psycopg3 驅動），HTTP 呼叫使用 `httpx.AsyncClient`
3. **向量格式**：pgvector 欄位寫入時直接傳 Python `list[float]`，無需手動序列化為字串
4. **OpenCC 保證**：所有 LLM 輸出的中文均經 `opencc s2t` 確保正體中文
5. **不修改現有 DB 結構**：欄位定義依照既有 Schema，不新增或改動欄位
6. **全參數化查詢**：條件篩選使用 `.in_(list)` 等 SQLAlchemy 方法，自動綁定參數，杜絕 SQL Injection
7. **DictionaryMatcher 動態篩選**：`官方正規詞彙` 6000+ 筆字典透過 Aho-Corasick 自動機，掃描輸入文本後僅注入命中術語（~10-50 筆），避免塞爆 LLM context window
8. **優雅的斷線中斷機制**：使用 `monitor_disconnect` (依賴 `await request.receive()`) 監聽 ASGI 斷線事件，結合 `run_interruptible` 封裝長耗時任務 (如向量檢索與 LLM 串流生成)，確保客戶端斷線時能立刻拋出 `asyncio.CancelledError`，釋放系統資源並阻止無效的資料庫寫入 (回傳 499 狀態碼)
