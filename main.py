"""
N530_FoodExportationQA — 食品輸銷問答 API
Main application entry point.
"""
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.logging_config import setup_logging
from app.database import init_engine, close_engine
from app.routers import qa, translate

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize engine on startup, close on shutdown."""
    setup_logging()
    logger.info("啟動 N530_FoodExportationQA ...")
    await init_engine()
    logger.info("應用程式啟動完成")
    yield
    logger.info("關閉 N530_FoodExportationQA ...")
    await close_engine()
    logger.info("應用程式關閉完成")


_OPENAPI_TAGS = [
    {
        "name": "QA",
        "description": (
            "食品輸銷問答核心端點。\n\n"
            "- **`POST /ask`** — 以 JSON body 提交問題，執行完整 RAG 管線（翻譯 → 意圖分類 → 雙軌檢索 → LLM 雙語生成）。\n"
            "- **`GET /ask`** — 以 Query String 提交問題，功能同 POST，適合瀏覽器直接測試。\n"
            "- **`POST /ingest`** — 指定問卷主檔主鍵，觸發切塊向量化並寫入資料庫，供後續檢索使用。\n"
            "- **`POST /breakdown`** — 上傳問卷原始檔案（PDF/DOCX/XLSX 等），透過 LLM 萃取結構化題號與英文題目清單。"
        ),
    },
    {
        "name": "Translation",
        "description": (
            "術語約束翻譯端點。\n\n"
            "- **`POST /re-translate`** — 將中文文本翻譯為英文，翻譯過程中自動比對「官方正規詞彙」資料庫，"
            "確保專業術語以官方核定英文名詞輸出。"
        ),
    },
    {
        "name": "Health",
        "description": "服務健康狀態檢查端點，可用於 Kubernetes liveness probe 或負載平衡器心跳偵測。",
    },
]

app = FastAPI(
    title="N530 食品輸銷問答 API",
    description=(
        "## 系統簡介\n\n"
        "基於 **RAG（Retrieval-Augmented Generation）** 架構的食品輸銷雙語問答系統。\n"
        "整合問卷歷史資料庫（Track A）與知識文獻庫（Track B），"
        "透過向量相似度搜尋與 LLM 生成，回傳負責單位、中英雙語解答及參考來源。\n\n"
        "## 技術架構\n\n"
        "| 元件 | 說明 |\n"
        "|------|------|\n"
        "| Embedding | `intfloat/multilingual-e5-large-instruct`（1024 維） |\n"
        "| LLM | `gpt-oss-20b` 雙語生成 |\n"
        "| 向量資料庫 | PostgreSQL + pgvector（cosine distance） |\n"
        "| 意圖分類 | Aho-Corasick 雙層 + LLM 兜底 |\n"
        "| 術語比對 | DictionaryMatcher（Aho-Corasick，6000+ 筆官方正規詞彙） |\n\n"
        "## 文件連結\n\n"
        "- Swagger UI：[/docs](/docs)\n"
        "- ReDoc：[/redoc](/redoc)\n"
        "- OpenAPI JSON：[/openapi.json](/openapi.json)"
    ),
    version="1.0.0",
    openapi_tags=_OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    """Log every incoming request with method, path, status code and elapsed time."""
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except RuntimeError as exc:
        if str(exc) == "No response returned." and await request.is_disconnected():
            logger.info("客戶端已斷線，終止請求 (路徑: %s)", request.url.path)
            # 499 Client Closed Request (Nginx standard)
            return Response(status_code=499)
        raise
        
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s — %s  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


app.include_router(qa.router)
app.include_router(translate.router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, log_level="info", reload=True)