-- ============================================================
-- Migration 003: 問卷題目切塊 — 重構為單向量欄位
-- 舊結構（雙向量：題目向量 + 回覆向量）→ 新結構（單向量：embedding + chunk_source 標記）
-- ============================================================

-- ── Step 1: 刪除舊表（包含所有舊切塊資料）──────────────────────────────────
DROP TABLE IF EXISTS public."問卷題目切塊";

-- ── Step 2: 建立新問卷題目切塊 ────────────────────────────────────────────────
-- chunk_source 標記切塊來源：'question'（題目文字）或 'answer'（回覆文字）
-- 同一筆問卷題目檔主鍵下，question 與 answer 的切塊各自獨立，切塊索引各自從 0 開始
CREATE TABLE IF NOT EXISTS public."問卷題目切塊" (
    "主鍵"           uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
    "問卷題目檔主鍵" varchar(40)  NOT NULL,
    "chunk_source"   varchar(50)  NOT NULL,    -- 'question' | 'answer'
    "chunk_text"     text,
    "embedding"      vector(1024),
    "切塊索引"       integer      NOT NULL,
    "詞元數量"       integer
);

COMMENT ON TABLE  public."問卷題目切塊"               IS '問卷題目的滑動視窗切塊 + 1024-dim 向量（由 /api/v1/qa/ingest 生成）';
COMMENT ON COLUMN public."問卷題目切塊"."chunk_source" IS 'question=題目文字, answer=回覆文字';
COMMENT ON COLUMN public."問卷題目切塊"."embedding"    IS 'intfloat/multilingual-e5-large-instruct 1024-dim cosine vector';

-- ── Step 3: 建立索引 ──────────────────────────────────────────────────────────
-- HNSW 向量索引（cosine distance）
CREATE INDEX IF NOT EXISTS idx_問卷題目切塊_embedding
    ON public."問卷題目切塊"
    USING hnsw ("embedding" vector_cosine_ops);

-- 外鍵關聯索引（JOIN 問卷題目檔時使用）
CREATE INDEX IF NOT EXISTS idx_問卷題目切塊_題目檔主鍵
    ON public."問卷題目切塊" ("問卷題目檔主鍵");

-- ── 完成 ──────────────────────────────────────────────────────────────────────
-- 後續步驟：
--   重新執行 POST /api/v1/qa/ingest 對所有問卷主鍵重新切塊向量化
