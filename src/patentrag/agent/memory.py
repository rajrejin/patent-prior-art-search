"""SQLite session memory: cached query results and agent traces."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def query_hash(query: str, **params: Any) -> str:
    blob = json.dumps({"q": " ".join(query.split()), **params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class Memory:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT, corpus_hash TEXT, kind TEXT, "
            "payload TEXT, created REAL, PRIMARY KEY (key, corpus_hash, kind))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, query_hash TEXT, "
            "created REAL, status TEXT, steps TEXT, report TEXT)"
        )
        self.db.commit()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def get(self, key: str, corpus_hash: str, kind: str) -> Any | None:
        with self._lock:
            row = self.db.execute(
                "SELECT payload FROM cache WHERE key=? AND corpus_hash=? AND kind=?",
                (key, corpus_hash, kind),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, corpus_hash: str, kind: str, payload: Any) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                (key, corpus_hash, kind, json.dumps(payload, default=str), time.time()),
            )
            self.db.commit()

    def save_trace(
        self,
        qhash: str,
        status: str,
        steps: list[dict[str, Any]],
        report: Any,
        trace_id: str | None = None,
    ) -> str:
        tid = trace_id or uuid.uuid4().hex[:16]
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?)",
                (
                    tid,
                    qhash,
                    time.time(),
                    status,
                    json.dumps(steps, default=str),
                    json.dumps(report, default=str),
                ),
            )
            self.db.commit()
        return tid

    def load_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.db.execute(
                "SELECT id, query_hash, created, status, steps, report FROM traces WHERE id=?",
                (trace_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "query_hash": row[1],
            "created": row[2],
            "status": row[3],
            "steps": json.loads(row[4]),
            "report": json.loads(row[5]),
        }
