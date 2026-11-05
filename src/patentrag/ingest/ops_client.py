"""EPO OPS client: OAuth, polite throttling, retries, quota guard, gzip response cache.

Secrets are never logged. Cached files store only the request path/params and the
response body, never request headers.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import random
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from patentrag.config import Config, Secrets, load_config, load_secrets
from patentrag.ingest.credentials import AUTH_URL, basic_auth_header

log = logging.getLogger(__name__)

BASE = "https://ops.epo.org/3.2/rest-services"
SERVICE_RE = re.compile(r"(\w+)=(green|yellow|red|black):(\d+)", re.IGNORECASE)
# per-minute request budgets before the first throttling header arrives (conservative)
DEFAULT_PER_MIN = {"search": 20, "inpadoc": 40, "retrieval": 100, "images": 5, "other": 100}
MAX_ATTEMPTS = 10


class QuotaExceeded(RuntimeError):
    """The self-imposed weekly download budget has been reached."""


class OpsError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"OPS HTTP {status}: {message}")
        self.status = status


def week_start(now: datetime) -> str:
    """Monday 00:00 GMT of the week containing `now`, as an ISO date."""
    d = now.astimezone(UTC).date()
    return (d - timedelta(days=d.weekday())).isoformat()


class QuotaState:
    """Bytes downloaded in the current EPO week, persisted to fetch_state.json."""

    def __init__(self, path: Path, weekly_bytes: int, stop_fraction: float):
        self.path = path
        self.limit = int(weekly_bytes * stop_fraction)
        self.state: dict[str, Any] = {"week_start": "", "bytes": 0, "requests": 0}
        if path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))
        self._roll()

    def _roll(self, now: datetime | None = None) -> None:
        ws = week_start(now or datetime.now(UTC))
        if self.state.get("week_start") != ws:
            self.state.update({"week_start": ws, "bytes": 0, "requests": 0})

    @property
    def used(self) -> int:
        return int(self.state["bytes"])

    def check(self, now: datetime | None = None) -> None:
        self._roll(now)
        if self.used >= self.limit:
            self.save()
            raise QuotaExceeded(
                f"{self.used} of {self.limit} budgeted bytes used in week "
                f"{self.state['week_start']}; rerun `patentrag fetch` after the quota "
                "resets (Monday 00:00 GMT)."
            )

    def add(self, nbytes: int) -> None:
        self.state["bytes"] = self.used + nbytes
        self.state["requests"] = int(self.state.get("requests", 0)) + 1
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")


def cache_key(endpoint: str, params: dict[str, Any] | None) -> str:
    blob = json.dumps({"e": endpoint, "p": params or {}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class OpsClient:
    def __init__(
        self,
        cfg: Config | None = None,
        secrets: Secrets | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ):
        self.cfg = cfg or load_config()
        self.secrets = secrets or load_secrets()
        self.raw_dir = self.cfg.path(self.cfg.data.raw_dir)
        self.quota = QuotaState(
            self.cfg.path(self.cfg.data.fetch_state),
            self.cfg.data.weekly_quota_bytes,
            self.cfg.data.quota_stop_fraction,
        )
        self._sleep = sleep
        self._http = httpx.Client(timeout=60, transport=transport)
        self._token = ""
        self._token_expiry = 0.0
        self._next_ok: dict[str, float] = {}
        self._per_min: dict[str, int] = dict(DEFAULT_PER_MIN)
        self.network_requests = 0
        self.cache_hits = 0

    # -- auth ---------------------------------------------------------------
    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        r = self._http.post(
            AUTH_URL,
            headers={
                "Authorization": basic_auth_header(self.secrets.ops_key, self.secrets.ops_secret)
            },
            data={"grant_type": "client_credentials"},
        )
        if r.status_code != 200:
            raise OpsError(r.status_code, "token request failed")
        body = r.json()
        self._token = body["access_token"]
        self._token_expiry = time.time() + float(body.get("expires_in", 1200))
        return self._token

    # -- cache --------------------------------------------------------------
    def _cache_path(self, endpoint: str, params: dict[str, Any] | None) -> Path:
        parts = endpoint.strip("/").split("/")
        sub = (
            "search"
            if parts[-1] == "search"
            else parts[-1]
            if parts[0] == "published-data"
            else parts[0]
        )
        key = cache_key(endpoint, params)
        return self.raw_dir / sub / key[:2] / f"{key}.json.gz"

    def cached(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        p = self._cache_path(endpoint, params)
        if not p.exists():
            return None
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        return data

    def _store(self, endpoint: str, params: dict[str, Any] | None, rec: dict[str, Any]) -> None:
        p = self._cache_path(endpoint, params)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(rec, fh)
        tmp.replace(p)

    # -- throttling ---------------------------------------------------------
    @staticmethod
    def service_of(endpoint: str) -> str:
        if endpoint.startswith("/family/"):
            return "inpadoc"
        if endpoint.endswith("/search"):
            return "search"
        return "retrieval"

    def _respect_throttle(self, headers: httpx.Headers, service: str) -> None:
        """Pace each service by the per-minute budget OPS announces in X-Throttling-Control."""
        ctl = headers.get("x-throttling-control", "")
        for name, _state, limit in SERVICE_RE.findall(ctl):
            self._per_min[name.lower()] = max(int(limit), 1)
        per_min = self._per_min.get(service, 20)
        interval = 60.0 / per_min * 1.6  # 60 percent safety margin
        self._next_ok[service] = time.time() + interval

    # -- main request -------------------------------------------------------
    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        range_header: str | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any] | None:
        """Cached GET. Returns the record, or None for a tolerated 404."""
        cache_params = dict(params or {})
        if range_header:
            cache_params["_range"] = range_header
        hit = self.cached(endpoint, cache_params)
        if hit is not None:
            self.cache_hits += 1
            return None if hit["status"] in (400, 404) else hit
        self.quota.check()
        attempt = 0
        while True:
            service = self.service_of(endpoint)
            wait = self._next_ok.get(service, 0.0) - time.time()
            if wait > 0:
                self._sleep(wait)
            headers = {
                "Authorization": f"Bearer {self._ensure_token()}",
                "Accept": "application/xml",
            }
            if range_header:
                headers["X-OPS-Range"] = range_header
            try:
                r = self._http.get(BASE + endpoint, params=params, headers=headers)
            except httpx.HTTPError as exc:
                attempt += 1
                if attempt > MAX_ATTEMPTS:
                    raise OpsError(0, f"network error {type(exc).__name__}") from exc
                self._backoff(attempt, None)
                continue
            self.network_requests += 1
            self.quota.add(len(r.content))
            self._respect_throttle(r.headers, service)
            for h in ("x-individualquotaperhour-used", "x-registeredquotaperweek-used"):
                if h in r.headers:
                    self.quota.state[h] = r.headers[h]
            if r.status_code in (200, 404) or (r.status_code == 400 and allow_404):
                rec = {
                    "endpoint": endpoint,
                    "params": cache_params,
                    "status": r.status_code,
                    "body": r.text,
                    "fetched": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                self._store(endpoint, cache_params, rec)
                if r.status_code in (400, 404):
                    if allow_404:
                        return None
                    raise OpsError(404, f"not found: {endpoint}")
                return rec
            if r.status_code == 401 and attempt < 2:
                self._token = ""
                attempt += 1
                continue
            reason = r.headers.get("x-rejection-reason", "")
            if r.status_code == 403 and reason.startswith("RegisteredQuota"):
                self.quota.save()
                raise QuotaExceeded(f"OPS rejected the request: {reason}")
            retryable = r.status_code in (429, 500, 502, 503, 504) or r.status_code == 403
            if not retryable or attempt >= MAX_ATTEMPTS:
                raise OpsError(r.status_code, r.text[:200].replace("\n", " "))
            attempt += 1
            self._backoff(
                attempt,
                r.headers.get("retry-after"),
                r.status_code,
                r.headers.get("x-rejection-reason", ""),
            )
            self.quota.check()

    def _backoff(
        self, attempt: int, retry_after: str | None, status: int = 0, reason: str = ""
    ) -> None:
        if retry_after and retry_after.isdigit():
            # OPS sends Retry-After in milliseconds (for example 900000 for 15 minutes).
            raw = float(retry_after)
            delay = raw / 1000.0 if raw > 3600 else raw
            delay = min(delay, 1800.0)
        else:
            delay = min(300.0, 2.0**attempt) + random.uniform(0, 1.0)  # noqa: S311
        log.warning(
            "OPS backoff %.1fs (attempt %d, HTTP %s, reason %s)",
            delay,
            attempt,
            status or "n/a",
            reason or "n/a",
        )
        self._sleep(delay)

    # -- typed helpers ------------------------------------------------------
    def search(self, cql: str, start: int, end: int) -> str:
        rec = self.get("/published-data/search", {"q": cql}, range_header=f"{start}-{end}")
        assert rec is not None
        return str(rec["body"])

    def _doc(self, kind: str, ref: str) -> str | None:
        rec = self.get(f"/published-data/publication/docdb/{ref}/{kind}", allow_404=True)
        return None if rec is None else str(rec["body"])

    def biblio(self, ref: str) -> str | None:
        return self._doc("biblio", ref)

    def claims(self, ref: str) -> str | None:
        return self._doc("claims", ref)

    def description(self, ref: str) -> str | None:
        return self._doc("description", ref)

    def family(self, ref: str) -> str | None:
        rec = self.get(f"/family/publication/docdb/{ref}", allow_404=True)
        return None if rec is None else str(rec["body"])

    def close(self) -> None:
        self._http.close()
