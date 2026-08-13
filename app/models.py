"""
SQLAlchemy 2.0 Core table definitions for all 10 database tables.
Uses Table() + MetaData — NOT ORM DeclarativeBase.

Dual-backend support (PostgreSQL + SQL Server 2025):
  - Column types use .with_variant() so the correct DDL/binding is used per dialect.
  - Schema is read from DB_SCHEMA env var ("public" for PG, "dbo" for MSSQL).
"""
import os
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    DateTime,
    Text,
    func,
    text,
)
from sqlalchemy.dialects import mssql
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from pgvector.sqlalchemy import Vector

from app.db_dialect import MssqlVector, MssqlArrayString

# Schema is overrideable via DB_SCHEMA env var.
# PostgreSQL default: "public"; SQL Server default: "dbo"
_DB_SCHEMA = os.environ.get("DB_SCHEMA", "public")

metadata = MetaData()

# ─── 問卷主檔 ───────────────────────────────────────────────
問卷主檔 = Table(
    "問卷主檔",
    metadata,
    Column("主鍵", String(40), primary_key=True),
    Column("輸出國家", String(40)),
    Column("輸出品項", String(40)),
    Column("問卷名稱", String(200)),
    Column("發文日期", Date),
    Column("發文文號", String(200)),
    Column("主辦機關", String(200)),
    Column("協辦機關", String(200)),
    Column("是否刪除", Integer),
    Column("刪除人員", String(50)),
    Column("刪除日期", DateTime),
    Column("建立人員", String(50)),
    Column("建立日期", DateTime),
    Column("修改人員", String(50)),
    Column("修改日期", DateTime),
    schema=_DB_SCHEMA,
)

# ─── 問卷題目檔 ──────────────────────────────────────────────
問卷題目檔 = Table(
    "問卷題目檔",
    metadata,
    Column("主鍵", String(40), primary_key=True),
    Column("問卷主檔主鍵", String(40)),
    Column("上層題目主鍵", String(40)),
    Column("題目序號", String(20)),
    Column("題目", String, nullable=False),
    Column("回覆", String),
    Column("排序", Integer, nullable=False),
    Column("發文日期", Date),
    Column("是否刪除", Integer),
    Column("刪除人員", String(50)),
    Column("刪除日期", DateTime),
    Column("建立人員", String(50)),
    Column("建立日期", DateTime, nullable=False),
    Column("修改人員", String(50)),
    Column("修改日期", DateTime),
    schema=_DB_SCHEMA,
)

# ─── 問卷題目切塊 ─────────────────────────────────────────────
問卷題目切塊 = Table(
    "問卷題目切塊",
    metadata,
    Column("主鍵",
           UUID(as_uuid=True).with_variant(mssql.UNIQUEIDENTIFIER(), "mssql"),
           primary_key=True, default=uuid.uuid4),
    Column("問卷題目檔主鍵", String(40)),
    Column("chunk_source", String(50), nullable=False),  # 'question' | 'answer'
    Column("chunk_text", String),
    Column("embedding",
           Vector(1024).with_variant(MssqlVector(1024), "mssql")),
    Column("切塊索引", Integer, nullable=False),
    Column("詞元數量", Integer),
    schema=_DB_SCHEMA,
)

# ─── 問卷附件檔 ──────────────────────────────────────────────
問卷附件檔 = Table(
    "問卷附件檔",
    metadata,
    Column("主鍵", String(40), primary_key=True),
    Column("問卷主檔主鍵", String(40)),
    Column("檔案名稱", String(200)),
    Column("是否刪除", Integer, nullable=False),
    Column("刪除人員", String(50)),
    Column("刪除日期", DateTime),
    Column("建立人員", String(50)),
    Column("建立日期", DateTime, nullable=False),
    Column("修改人員", String(50)),
    Column("修改日期", DateTime),
    Column("檔案路徑", String(500)),
    schema=_DB_SCHEMA,
)

# ─── 問卷原始檔 ──────────────────────────────────────────────
問卷原始檔 = Table(
    "問卷原始檔",
    metadata,
    Column("主鍵", String(40), primary_key=True),
    Column("問卷主檔主鍵", String(40)),
    Column("檔案名稱", String(200)),
    Column("是否刪除", Integer, nullable=False),
    Column("刪除人員", String(50)),
    Column("刪除日期", DateTime),
    Column("建立人員", String(50)),
    Column("建立日期", DateTime, nullable=False),
    Column("修改人員", String(50)),
    Column("修改日期", DateTime),
    Column("檔案路徑", String(500)),
    schema=_DB_SCHEMA,
)

# ─── 單位對照表 ──────────────────────────────────────────────
單位對照表 = Table(
    "單位對照表",
    metadata,
    Column("機關", String),
    Column("單位", String),
    Column("分工關鍵字", String),
    Column("擴充關鍵字",
           ARRAY(String).with_variant(MssqlArrayString(), "mssql")),
    schema=_DB_SCHEMA,
)

# ─── 官方正規詞彙 ─────────────────────────────────────────────
官方正規詞彙 = Table(
    "官方正規詞彙",
    metadata,
    Column("中文字詞", String),
    Column("英文字詞", String),
    schema=_DB_SCHEMA,
)

# ─── 知識文獻主檔 ──────────────────────────────────────────────
# 支援文獻類型：REGULATION（法規）、GUIDELINE（作業指引）、QA（問答集）
知識文獻主檔 = Table(
    "知識文獻主檔",
    metadata,
    Column("主鍵", String(40), primary_key=True),
    Column("文獻名稱", String(500), nullable=False),
    Column("文獻類型", String(20), nullable=False),
    Column("原始檔案路徑", String(500)),
    Column("是否刪除", Integer, server_default=text("0")),
    Column("建立日期", DateTime, nullable=False, server_default=func.now()),
    schema=_DB_SCHEMA,
)

# ─── 知識文獻節點檔 ───────────────────────────────────────────────
# LLM 從 Markdown 萃取的結構化節點（章節 / 條號 / Q&A）
知識文獻節點檔 = Table(
    "知識文獻節點檔",
    metadata,
    Column("主鍵", String(40), primary_key=True),
    Column("文獻主檔主鍵", String(40), nullable=False),
    Column("節點標題路徑", String(500), nullable=False),
    Column("節點內容", String, nullable=False),
    Column("排序索引", Integer, nullable=False, server_default=text("1")),
    Column("是否刪除", Integer, server_default=text("0")),
    Column("建立日期", DateTime, nullable=False, server_default=func.now()),
    schema=_DB_SCHEMA,
)

# ─── 知識文獻切塊檔 ───────────────────────────────────────────────
# 節點內容的滑動視窗切塊 + 1024-dim 向量，由 tools/ingest_agent.py 自動管理
知識文獻切塊檔 = Table(
    "知識文獻切塊檔",
    metadata,
    Column("主鍵",
           UUID(as_uuid=True).with_variant(mssql.UNIQUEIDENTIFIER(), "mssql"),
           primary_key=True, default=uuid.uuid4),
    Column("文獻節點檔主鍵", String(40), nullable=False),
    Column("切塊內容", String),
    Column("內容向量",
           Vector(1024).with_variant(MssqlVector(1024), "mssql")),
    Column("切塊索引", Integer, nullable=False),
    Column("詞元數量", Integer),
    schema=_DB_SCHEMA,
)

# ─── 查詢紀錄 ──────────────────────────────────────────────
# 每次 /qa/ask 呼叫寫入一筆，記錄完整管線上下文，可追溯提供給 LLM 的所有輸入
查詢紀錄 = Table(
    "查詢紀錄",
    metadata,
    Column("主鍵",
           UUID(as_uuid=True).with_variant(mssql.UNIQUEIDENTIFIER(), "mssql"),
           primary_key=True, default=uuid.uuid4),
    Column("查詢時間", DateTime, nullable=False, server_default=func.now()),
    Column("原始問題", Text, nullable=False),
    Column("是否中文",
           Boolean().with_variant(mssql.BIT(), "mssql"),
           nullable=False),
    Column("中文譯文", Text),
    Column("意圖分類方式", String(20)),         # 'aho-corasick' | 'llm_fallback' | 'none'
    Column("負責機關", String(200)),
    Column("負責單位", String(200)),
    Column("命中術語",
           JSONB().with_variant(mssql.JSON(), "mssql")),   # {中文: 英文, ...}
    Column("enriched_query", Text),
    Column("similarity_threshold", Float),
    Column("top_n", Integer),
    Column("問卷命中數", Integer),
    Column("知識命中數", Integer),
    Column("參考來源",
           JSONB().with_variant(mssql.JSON(), "mssql")),
    Column("知識參考來源",
           JSONB().with_variant(mssql.JSON(), "mssql")),
    Column("raw_context", Text),
    Column("knowledge_context", Text),
    Column("dictionary_xml", Text),
    Column("llm_prompt", Text),
    Column("llm_output", Text),
    Column("英文回覆", Text),
    Column("中文回覆", Text),
    Column("耗時毫秒", Float),
    Column("錯誤訊息", Text),
    schema=_DB_SCHEMA,
)
