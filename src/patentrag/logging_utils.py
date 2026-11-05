"""JSON logging with request ids. Query text is never logged unless debug is on."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_SENSITIVE_KEYS = {"authorization", "client_secret", "consumer_secret", "password", "api_key"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update({k: v for k, v in extra.items() if k.lower() not in _SENSITIVE_KEYS})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def debug_queries_enabled() -> bool:
    return os.environ.get("PATENTRAG_DEBUG_QUERIES", "") == "1"


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        root.addHandler(h)
    # httpx logs full URLs at INFO; keep it quiet so nothing sensitive leaks.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
