-- migrations/005_mssql_schema.sql
-- SQL Server 2025 schema for N530 FoodExportationQA
-- Run this once against your SQL Server [fes] database as a user with DDL rights.
-- All tables reside in [dbo] schema (configure DB_SCHEMA=dbo in .env).
-- Requires: SQL Server 2025 preview (or later) with native VECTOR type support.

USE [fes];
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 1. 問卷主檔
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[問卷主檔]', N'U') IS NULL
CREATE TABLE [dbo].[問卷主檔] (
    [主鍵]     NVARCHAR(40),
    [輸出國家]  NVARCHAR(40),
    [輸出品項]  NVARCHAR(40),
    [問卷名稱]  NVARCHAR(200),
    [發文日期]  DATE,
    [發文文號]  NVARCHAR(200),
    [主辦機關]  NVARCHAR(200),
    [協辦機關]  NVARCHAR(200),
    [是否刪除]  INT,
    [刪除人員]  NVARCHAR(50),
    [刪除日期]  DATETIME2,
    [建立人員]  NVARCHAR(50),
    [建立日期]  DATETIME2,
    [修改人員]  NVARCHAR(50),
    [修改日期]  DATETIME2
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 2. 問卷題目檔
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[問卷題目檔]', N'U') IS NULL
CREATE TABLE [dbo].[問卷題目檔] (
    [主鍵]          NVARCHAR(40),
    [問卷主檔主鍵]   NVARCHAR(40),
    [上層題目主鍵]   NVARCHAR(40),
    [題目序號]       NVARCHAR(20),
    [題目]           NVARCHAR(MAX) NOT NULL,
    [回覆]           NVARCHAR(MAX),
    [排序]           INT           NOT NULL,
    [發文日期]       DATE,
    [是否刪除]       INT,
    [刪除人員]       NVARCHAR(50),
    [刪除日期]       DATETIME2,
    [建立人員]       NVARCHAR(50),
    [建立日期]       DATETIME2     NOT NULL,
    [修改人員]       NVARCHAR(50),
    [修改日期]       DATETIME2
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 3. 問卷題目切塊
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[問卷題目切塊]', N'U') IS NULL
CREATE TABLE [dbo].[問卷題目切塊] (
    [主鍵]           UNIQUEIDENTIFIER NOT NULL DEFAULT NEWID() PRIMARY KEY,
    [問卷題目檔主鍵]  NVARCHAR(40),
    [chunk_source]   NVARCHAR(50)     NOT NULL,   -- 'question' | 'answer'
    [chunk_text]     NVARCHAR(MAX),
    [embedding]      VECTOR(1024),
    [切塊索引]        INT              NOT NULL,
    [詞元數量]        INT
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 4. 問卷附件檔
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[問卷附件檔]', N'U') IS NULL
CREATE TABLE [dbo].[問卷附件檔] (
    [主鍵]         NVARCHAR(40),
    [問卷主檔主鍵]  NVARCHAR(40),
    [檔案名稱]      NVARCHAR(200),
    [是否刪除]      INT           NOT NULL,
    [刪除人員]      NVARCHAR(50),
    [刪除日期]      DATETIME2,
    [建立人員]      NVARCHAR(50),
    [建立日期]      DATETIME2     NOT NULL,
    [修改人員]      NVARCHAR(50),
    [修改日期]      DATETIME2,
    [檔案路徑]      NVARCHAR(500)
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 5. 問卷原始檔
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[問卷原始檔]', N'U') IS NULL
CREATE TABLE [dbo].[問卷原始檔] (
    [主鍵]         NVARCHAR(40),
    [問卷主檔主鍵]  NVARCHAR(40),
    [檔案名稱]      NVARCHAR(200),
    [是否刪除]      INT           NOT NULL,
    [刪除人員]      NVARCHAR(50),
    [刪除日期]      DATETIME2,
    [建立人員]      NVARCHAR(50),
    [建立日期]      DATETIME2     NOT NULL,
    [修改人員]      NVARCHAR(50),
    [修改日期]      DATETIME2,
    [檔案路徑]      NVARCHAR(500)
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 6. 單位對照表
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[單位對照表]', N'U') IS NULL
CREATE TABLE [dbo].[單位對照表] (
    [機關]      NVARCHAR(MAX),
    [單位]      NVARCHAR(MAX),
    [分工關鍵字] NVARCHAR(MAX),
    [擴充關鍵字] NVARCHAR(MAX)    -- stores JSON array: ["alias1","alias2",...]
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 7. 官方正規詞彙
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[官方正規詞彙]', N'U') IS NULL
CREATE TABLE [dbo].[官方正規詞彙] (
    [中文字詞] NVARCHAR(MAX),
    [英文字詞] NVARCHAR(MAX)
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 8. 知識文獻主檔
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[知識文獻主檔]', N'U') IS NULL
CREATE TABLE [dbo].[知識文獻主檔] (
    [主鍵]          NVARCHAR(40)  NOT NULL PRIMARY KEY,
    [文獻名稱]       NVARCHAR(500) NOT NULL,
    [文獻類型]       NVARCHAR(20)  NOT NULL
        CONSTRAINT chk_文獻類型 CHECK ([文獻類型] IN ('REGULATION', 'GUIDELINE', 'QA')),
    [原始檔案路徑]   NVARCHAR(500),
    [是否刪除]       INT           DEFAULT 0,
    [建立日期]       DATETIME2     NOT NULL DEFAULT GETDATE()
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 9. 知識文獻節點檔
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[知識文獻節點檔]', N'U') IS NULL
CREATE TABLE [dbo].[知識文獻節點檔] (
    [主鍵]         NVARCHAR(40)  NOT NULL PRIMARY KEY,
    [文獻主檔主鍵]  NVARCHAR(40)  NOT NULL,
    [節點標題路徑]  NVARCHAR(500) NOT NULL,
    [節點內容]      NVARCHAR(MAX) NOT NULL,
    [排序索引]      INT           NOT NULL DEFAULT 1,
    [是否刪除]      INT           DEFAULT 0,
    [建立日期]      DATETIME2     NOT NULL DEFAULT GETDATE()
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 10. 知識文獻切塊檔
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[知識文獻切塊檔]', N'U') IS NULL
CREATE TABLE [dbo].[知識文獻切塊檔] (
    [主鍵]           UNIQUEIDENTIFIER NOT NULL DEFAULT NEWID() PRIMARY KEY,
    [知識文獻節點檔主鍵]  NVARCHAR(40)     NOT NULL,
    [切塊內容]        NVARCHAR(MAX),
    [內容向量]        VECTOR(1024),
    [切塊索引]        INT              NOT NULL,
    [詞元數量]        INT
);
GO

-- ──────────────────────────────────────────────────────────────────────────────
-- 11. 查詢紀錄  (optional — only if you log queries)
-- ──────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID(N'[dbo].[查詢紀錄]', N'U') IS NULL
CREATE TABLE [dbo].[查詢紀錄] (
    [主鍵]          UNIQUEIDENTIFIER NOT NULL DEFAULT NEWID() PRIMARY KEY,
    [查詢文字]       NVARCHAR(MAX),
    [是否中文]       BIT              NOT NULL,
    [命中術語]       NVARCHAR(MAX),    -- JSON
    [參考來源]       NVARCHAR(MAX),    -- JSON
    [知識參考來源]   NVARCHAR(MAX),    -- JSON
    [建立日期]       DATETIME2        NOT NULL DEFAULT GETDATE()
);
GO
