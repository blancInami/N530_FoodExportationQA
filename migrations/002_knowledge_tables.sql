-- ============================================================
-- Migration 002: 知識文獻庫 Schema
-- 將舊法規三表替換為泛用型知識文獻三表
-- 支援文獻類型：REGULATION（法規）、GUIDELINE（作業指引）、QA（問答集）
-- ============================================================

-- ── Step 1: 刪除舊法規三表（依外鍵順序逆向刪除）──────────────────────────
DROP TABLE IF EXISTS public."法規切塊檔";
DROP TABLE IF EXISTS public."法條內容檔";
DROP TABLE IF EXISTS public."法規主檔";

-- ── Step 2: 啟用 pgvector 擴展（若尚未啟用）────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Step 3: 建立知識文獻主檔 ─────────────────────────────────────────────────
-- 存放文獻的基本資訊：名稱、類型（REGULATION/GUIDELINE/QA）、原始檔案路徑
CREATE TABLE IF NOT EXISTS public."知識文獻主檔" (
    "主鍵"         varchar(40) PRIMARY KEY,
    "文獻名稱"     varchar(500) NOT NULL,
    "文獻類型"     varchar(20)  NOT NULL
                   CHECK ("文獻類型" IN ('REGULATION', 'GUIDELINE', 'QA')),
    "原始檔案路徑" varchar(500),
    "是否刪除"     integer      DEFAULT 0,
    "建立日期"     timestamp    NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public."知識文獻主檔"           IS '知識庫文獻基本資訊（人工匯入 or 工具匯入）';
COMMENT ON COLUMN public."知識文獻主檔"."文獻類型" IS 'REGULATION=法規, GUIDELINE=作業指引, QA=問答集';

-- ── Step 4: 建立文獻節點檔 ───────────────────────────────────────────────────
-- 存放 LLM 從 Markdown 萃取的結構化節點（章節 / 條號 / 問答）
CREATE TABLE IF NOT EXISTS public."文獻節點檔" (
    "主鍵"           varchar(40) PRIMARY KEY,
    "文獻主檔主鍵"   varchar(40) NOT NULL,
    "節點標題路徑"   varchar(500) NOT NULL,
    "節點內容"       text         NOT NULL,
    "排序索引"       integer      NOT NULL DEFAULT 1,
    "是否刪除"       integer      DEFAULT 0,
    "建立日期"       timestamp    NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public."文獻節點檔"                IS '文獻中的各節點（條號 / 章節 / Q&A）';
COMMENT ON COLUMN public."文獻節點檔"."節點標題路徑" IS '完整階層路徑，如 "第二章 > 第十五條" 或 "Q1: 申請資格為何"';

-- ── Step 5: 建立文獻切塊檔 ───────────────────────────────────────────────────
-- 節點內容的向量化切塊，由離線工具 tools/ingest_agent.py 自動管理
CREATE TABLE IF NOT EXISTS public."文獻切塊檔" (
    "主鍵"           uuid    PRIMARY KEY DEFAULT gen_random_uuid(),
    "文獻節點檔主鍵" varchar(40) NOT NULL,
    "切塊內容"       text,
    "內容向量"       vector(1024),
    "切塊索引"       integer NOT NULL,
    "詞元數量"       integer
);

COMMENT ON TABLE  public."文獻切塊檔"            IS '節點內容的滑動視窗切塊 + 1024-dim 向量（由 tools/ingest_agent.py 生成）';
COMMENT ON COLUMN public."文獻切塊檔"."內容向量" IS 'intfloat/multilingual-e5-large-instruct 1024-dim cosine vector';

-- ── Step 6: 建立索引 ─────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_文獻節點檔_主檔
    ON public."文獻節點檔" ("文獻主檔主鍵");

CREATE INDEX IF NOT EXISTS idx_文獻切塊檔_節點
    ON public."文獻切塊檔" ("文獻節點檔主鍵");

CREATE INDEX IF NOT EXISTS idx_文獻切塊檔_向量
    ON public."文獻切塊檔"
    USING hnsw ("內容向量" vector_cosine_ops);

-- ── 完成 ──────────────────────────────────────────────────────────────────────
-- 後續步驟：
--   1. 執行 tools/ingest_agent.py 匯入既有法規 PDF / SOP / QA 文件
--   2. POST /api/v1/qa/ask 即可使用新知識庫進行檢索
