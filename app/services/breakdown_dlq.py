"""Best-effort JSONL dead-letter storage for breakdown work units."""
import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_RAW_TEXT_LIMIT = 2_000


async def write_breakdown_dead_letter(
    *,
    stage: str,
    document_id: str,
    unit_index: int,
    total_units: int,
    error: BaseException | str,
    candidate_keys: list[str],
    diagnostics: dict[str, Any] | None = None,
    raw_payload: str = "",
) -> bool:
    """Append a sanitized failure record without affecting the calling pipeline."""
    settings = get_settings()
    if not settings.breakdown_dlq_enabled:
        return False

    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "needs_review",
        "stage": stage,
        "document_id": document_id,
        "unit_index": unit_index,
        "total_units": total_units,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "candidate_keys": candidate_keys,
        "diagnostics": diagnostics or {},
    }
    if raw_payload:
        record["raw_payload_sha256"] = hashlib.sha256(
            raw_payload.encode("utf-8")
        ).hexdigest()
        if settings.breakdown_dlq_include_raw_payload:
            record["raw_payload"] = raw_payload[:_RAW_TEXT_LIMIT]

    try:
        await asyncio.to_thread(_append_jsonl, Path(settings.breakdown_dlq_path), record)
    except OSError as exc:
        logger.error("Breakdown DLQ 寫入失敗：%s", exc, exc_info=True)
        return False
    return True


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Synchronously append one UTF-8 JSON object and create its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        stream.write("\n")