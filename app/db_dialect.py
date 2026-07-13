"""
Database dialect abstraction for dual-backend support (PostgreSQL + SQL Server 2025).

Provides:
  - MssqlVector(dim)    — SQLAlchemy UserDefinedType that renders as VECTOR(N) DDL
                           and transparently converts list[float] ↔ JSON string on bind/result.
  - MssqlArrayString()  — TypeDecorator over NVARCHAR(MAX) that converts list[str] ↔ JSON.
  - cosine_distance_expr(column, vector, dim) — backend-aware cosine distance expression:
        postgres → column.cosine_distance(vector)   (pgvector <=> operator)
        mssql    → VECTOR_DISTANCE('cosine', column, CAST(json, VECTOR(dim)))
"""
import json

from sqlalchemy import cast, func, literal
from sqlalchemy.dialects.mssql import NVARCHAR
from sqlalchemy.types import TypeDecorator, UserDefinedType


# ─── SQL Server VECTOR(N) type ───────────────────────────────────────────────

class MssqlVector(UserDefinedType):
    """
    Maps to SQL Server 2025 native VECTOR(dim) data type.
    Python side: list[float].  DB side: VECTOR binary, bound/read as JSON array string.
    """

    cache_ok = True

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def get_col_spec(self, **kw) -> str:  # type: ignore[override]
        return f"VECTOR({self.dim})"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return json.dumps(value)
            return value
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                return json.loads(value)
            return value
        return process


# ─── SQL Server Array-as-JSON type ───────────────────────────────────────────

class MssqlArrayString(TypeDecorator):
    """
    Stores Python list[str] as a JSON array in NVARCHAR(MAX) for SQL Server.
    Transparent to calling code: reads always return list[str] (never None).
    """

    impl = NVARCHAR(None)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []


# ─── Cosine distance expression factory ─────────────────────────────────────

def cosine_distance_expr(column, vector: list[float], dim: int = 1024):
    """
    Return a SQLAlchemy column expression for cosine distance between *column* and *vector*.

    PostgreSQL (pgvector):  column.cosine_distance(vector)  →  col <=> :v
    SQL Server 2025:        VECTOR_DISTANCE('cosine', col, CAST(:v AS VECTOR(dim)))
    """
    from app.config import get_settings  # deferred to avoid circular import at module level

    db_type = get_settings().db_type.lower()

    if db_type == "mssql":
        vec_json = json.dumps(vector)
        return func.VECTOR_DISTANCE(
            literal("cosine"),
            column,
            cast(literal(vec_json), MssqlVector(dim)),
        )
    else:
        # pgvector method — generates col <=> :param
        return column.cosine_distance(vector)
