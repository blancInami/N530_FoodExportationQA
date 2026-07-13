"""
Synchronous database connection factory for tools/ scripts.

Selects the correct driver based on the DB_TYPE environment variable:
  - "postgres"  → psycopg (sync) — existing behaviour
  - "mssql"     → pyodbc          — SQL Server 2025

Usage in tool scripts:
    from tools.db_utils import get_connection, PLACEHOLDER

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 WHERE col = {PLACEHOLDER}", (value,))
            rows = cur.fetchall()

IMPORTANT:
  - pyodbc uses "?" as the parameter placeholder.
  - psycopg uses "%s" as the parameter placeholder.
  - Always use the PLACEHOLDER constant for portability.
"""
import os
from contextlib import contextmanager
from typing import Generator


# ── Environment helpers ───────────────────────────────────────────────────────

def _db_type() -> str:
    return os.environ.get("DB_TYPE", "postgres").lower()


def _pg_dsn() -> str:
    host     = os.environ.get("PG_HOST", "127.0.0.1")
    port     = os.environ.get("PG_PORT", "5432")
    user     = os.environ.get("PG_USER", "postgres")
    password = os.environ.get("PG_PASSWORD", "postgres")
    db       = os.environ.get("PG_DB", "fes")
    return f"host={host} port={port} user={user} password={password} dbname={db}"


def _mssql_conn_str() -> str:
    host    = os.environ.get("MSSQL_HOST", "127.0.0.1")
    port    = os.environ.get("MSSQL_PORT", "1433")
    user    = os.environ.get("MSSQL_USER", "sa")
    pwd     = os.environ.get("MSSQL_PASSWORD", "")
    db      = os.environ.get("MSSQL_DB", "fes")
    driver  = os.environ.get("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")
    trust   = os.environ.get("MSSQL_TRUST_CERT", "no").lower() in ("1", "yes", "true")
    trust_s = ";TrustServerCertificate=yes" if trust else ""
    return (
        f"DRIVER={{{driver}}};SERVER={host},{port};DATABASE={db};"
        f"UID={user};PWD={pwd}{trust_s}"
    )


# ── Placeholder constant ──────────────────────────────────────────────────────

#: SQL parameter placeholder for use in execute() calls.
#: "?" for pyodbc (MSSQL); "%s" for psycopg (PostgreSQL).
PLACEHOLDER: str = "?" if _db_type() == "mssql" else "%s"


# ── Connection context manager ────────────────────────────────────────────────

@contextmanager
def get_connection() -> Generator:
    """
    Context manager that yields a synchronous DB-API connection.
    Commits on clean exit; rolls back and re-raises on exception.
    Always closes the connection on exit.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"INSERT INTO t (col) VALUES ({PLACEHOLDER})", (val,))
    """
    db = _db_type()

    if db == "mssql":
        import pyodbc  # type: ignore[import]
        conn = pyodbc.connect(_mssql_conn_str(), autocommit=False)
    else:
        import psycopg  # type: ignore[import]
        conn = psycopg.connect(_pg_dsn())

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_cursor(conn):
    """
    Return a cursor from *conn*, hiding the psycopg / pyodbc API difference.
    pyodbc cursors are not context managers in older versions; wrap if needed.
    """
    return conn.cursor()


def get_schema() -> str:
    """Return DB schema name from DB_SCHEMA env var (default: 'public' for PG, 'dbo' for MSSQL)."""
    default = "dbo" if _db_type() == "mssql" else "public"
    return os.environ.get("DB_SCHEMA", default)


def encode_array(value: list) -> object:
    """
    Encode a Python list for storage in the current dialect.
    - PostgreSQL (psycopg): returns value unchanged; psycopg handles list natively as array.
    - SQL Server (pyodbc):  returns JSON string (NVARCHAR(MAX) column stores JSON).
    """
    import json
    if _db_type() == "mssql":
        return json.dumps(value, ensure_ascii=False)
    return value
