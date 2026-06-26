-- Migration 004: 建立查詢紀錄資料表
-- 紀錄每次 /qa/ask 呼叫的完整上下文，供追溯提供給 LLM 的所有輸入。
-- 執行前確認已執行 001、002、003 migrations。

CREATE TABLE IF NOT EXISTS public."查詢紀錄" (
    "主鍵"                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    "查詢時間"              timestamp NOT NULL DEFAULT now(),
    "原始問題"              text NOT NULL,
    "是否中文"              boolean NOT NULL,
    "中文譯文"              text,
    -- 'aho-corasick' | 'llm_fallback' | 'none'
    "意圖分類方式"          varchar(20),
    "負責機關"              varchar(200),
    "負責單位"              varchar(200),
    -- Phase 1c filter_by_text 結果：{"中文術語": "英文術語", ...}
    "命中術語"              jsonb,
    "enriched_query"        text,
    "similarity_threshold"  float,
    "top_n"                 integer,
    "問卷命中數"            integer,
    "知識命中數"            integer,
    -- Track A 參考來源列表：[{"問卷主檔主鍵": ..., "問卷名稱": ..., ...}, ...]
    "參考來源"              jsonb,
    -- Track B 知識文獻來源列表：[{"文獻主檔主鍵": ..., "文獻名稱": ..., ...}, ...]
    "知識參考來源"          jsonb,
    -- 組裝給 LLM 的問卷歷史問答上下文
    "raw_context"           text,
    -- 組裝給 LLM 的知識文獻上下文
    "knowledge_context"     text,
    -- 注入 prompt 的術語字典 XML
    "dictionary_xml"        text,
    -- 完整 LLM prompt（可完整重現 LLM 的輸入）
    "llm_prompt"            text,
    -- LLM 原始回覆（含分隔線，未經解析）
    "llm_output"            text,
    "英文回覆"              text,
    "中文回覆"              text,
    "耗時毫秒"              float,
    -- 若管線過程發生例外，記錄錯誤訊息；正常完成則為 NULL
    "錯誤訊息"              text
);

-- 依查詢時間倒序排列的索引，供後續歷史查詢使用
CREATE INDEX IF NOT EXISTS idx_query_log_time
    ON public."查詢紀錄" ("查詢時間" DESC);
