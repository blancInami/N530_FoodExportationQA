"""
FastAPI dependencies for dependency injection.
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from app.config import Settings, get_settings
from app.database import get_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an AsyncSession from the session factory."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


def get_config() -> Settings:
    """Provide settings as a dependency."""
    return get_settings()
