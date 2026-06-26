"""
Async database engine using SQLAlchemy 2.0 Core + psycopg3 driver.
"""
import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_engine() -> None:
    """Initialize the async engine and session factory."""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        # Mask password in log output
        safe_dsn = settings.dsn.replace(f":{settings.pg_password}@", ":***@")
        logger.info(
            "初始化資料庫引擎：dsn=%s  pool_size=10  max_overflow=0",
            safe_dsn,
        )
        _engine = create_async_engine(
            settings.dsn,
            pool_size=10,
            max_overflow=0,
            pool_pre_ping=True,
            echo=False,
        )
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("資料庫引擎初始化完成")


async def close_engine() -> None:
    """Dispose the async engine."""
    global _engine, _session_factory
    if _engine is not None:
        logger.info("正在釋放資料庫引擎 ...")
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("資料庫引擎已釋放")


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory (must call init_engine first)."""
    if _session_factory is None:
        raise RuntimeError("Database engine not initialized. Call init_engine() first.")
    return _session_factory
