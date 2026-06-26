"""
Centralized logging configuration for N530_FoodExportationQA.
Call setup_logging() once at application startup (in lifespan).
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 1. 取得當前腳本絕對路徑，並定位至父目錄
# 第一個 .parent 退至腳本所在目錄，第二個 .parent 退至父目錄
_BASE_DIR = Path(__file__).resolve().parent.parent

# 2. 透過 / 運算子直接進行路徑拼接
_LOGS_DIR = _BASE_DIR / "logs"
_LOG_FILE = _LOGS_DIR / "app.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 10


def setup_logging() -> None:
    """
    Configure root logger with a consistent format.
    Log level is read from the LOG_LEVEL environment variable (default: INFO).
    Outputs to both stdout and a rotating file at logs/app.log.
    Third-party noisy loggers are suppressed to WARNING.
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    os.makedirs(_LOGS_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any existing handlers to avoid duplicates (e.g., uvicorn adds its own)
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("sqlalchemy.engine", "sqlalchemy.pool", "httpx", "httpcore", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("日誌系統初始化完成 — 等級=%s  檔案=%s", log_level_name, _LOG_FILE)