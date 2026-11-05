"""FastAPI service: /v1/health, /v1/ready, /v1/search, /v1/report (agent, synchronous),
/v1/report/jobs (agent, asynchronous: submit then poll), /v1/trace/{id}, /metrics."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.concurrency import run_in_threadpool

from api.schemas import ReportRequest, SearchRequest
from patentrag.config import Config, load_config
from patentrag.llm.client import LLMUnavailable
from patentrag.logging_utils import debug_queries_enabled, request_id_var, setup_logging
from patentrag.schemas import ErrorResponse, ReportResponse, SearchResponse
from patentrag.service import PatentService

log = logging.getLogger("api")


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "patentrag_requests_total", "Requests", ["path", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "patentrag_search_seconds",
            "Search latency",
            ["method"],
            registry=self.registry,
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
        )
        self.llm_latency = Histogram(
            "patentrag_report_seconds",
            "Report generation latency",
            registry=self.registry,
            buckets=(5, 15, 30, 60, 120, 300, 600),
        )
        self.retries = Counter(
            "patentrag_report_retries_total", "Report retries", registry=self.registry
        )
        self.faith_fail = Counter(
            "patentrag_faithfulness_failures_total",
            "Reports returned with a warning",
            registry=self.registry,
        )
        self.degraded = Counter(
            "patentrag_degraded_searches_total", "Graph fallbacks", registry=self.registry
        )


def create_app(service: PatentService | None = None, cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    setup_logging(os.environ.get("PATENTRAG_LOG_LEVEL", "INFO"))
    svc = service or PatentService(cfg)
    metrics = Metrics()
    llm_gate = asyncio.Semaphore(1)  # LLM calls are serialised
    jobs: dict[str, dict[str, Any]] = {}  # report jobs, newest last; at most cfg.api.max_jobs
    api_key = os.environ.get("PATENTRAG_API_KEY", "")
    app = FastAPI(title="patentrag", version="0.1.0", docs_url="/docs")
    app.state.service = svc
    app.state.metrics = metrics
    app.state.tasks = set()  # keeps references to running report jobs

    if cfg.api.cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"])

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        metrics.requests.labels(request.url.path, str(response.status_code)).inc()
        log.info(
            "request",
            extra={
                "fields": {
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            },
        )
        return response

    def err(status: int, error: str, detail: str = "") -> JSONResponse:
        body = ErrorResponse(error=error, detail=detail, request_id=request_id_var.get())
        return JSONResponse(status_code=status, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return err(422, "invalid_request", f"{loc}: {first.get('msg', 'invalid value')}")

    @app.exception_handler(HTTPException)
    async def http_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return err(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(LLMUnavailable)
    async def llm_handler(_: Request, exc: LLMUnavailable) -> JSONResponse:
        return err(503, "llm_unavailable", "The local language model is not reachable.")

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error: %s", type(exc).__name__)
        return err(500, "internal_error", "An internal error occurred.")

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    @app.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/ready")
    async def ready() -> JSONResponse:
        state: dict[str, Any] = await run_in_threadpool(svc.readiness)
        problems = []
        if not state.get("indexes"):
            problems.append("search indexes are missing (run `patentrag build-index`)")
        if not state.get("neo4j"):
            problems.append("graph database is unavailable (graph tools and graphrag degrade)")
        if not state.get("llm"):
            problems.append("language model is unavailable (reports cannot be generated)")
        status = 200 if not problems else 503
        content = {"ready": not problems, "problems": problems, **state}
        return JSONResponse(status_code=status, content=content)

    @app.post("/v1/search", response_model=SearchResponse, dependencies=[Depends(require_key)])
    async def search(req: SearchRequest) -> Any:
        if len(req.query) > cfg.api.max_query_chars or req.k > cfg.api.max_k:
            raise HTTPException(status_code=422, detail="query too long or k too large")
        if debug_queries_enabled():
            log.debug("search query: %s", req.query)
        t0 = time.perf_counter()
        try:
            res = await asyncio.wait_for(
                run_in_threadpool(svc.search, req.query, req.k, req.method),
                timeout=cfg.api.timeout_s,
            )
        except TimeoutError:
            return err(504, "timeout", "The search took too long.")
        metrics.latency.labels(res.method).observe(time.perf_counter() - t0)
        if res.degraded:
            metrics.degraded.inc()
        return res

    def observe(res: ReportResponse, seconds: float) -> None:
        metrics.llm_latency.observe(seconds)
        metrics.retries.inc(max(res.attempts - 1, 0))
        if res.warning:
            metrics.faith_fail.inc()

    @app.post("/v1/report", response_model=ReportResponse, dependencies=[Depends(require_key)])
    async def report(req: ReportRequest) -> Any:
        t0 = time.perf_counter()
        async with llm_gate:
            try:
                res = await asyncio.wait_for(
                    run_in_threadpool(svc.report, req.query), timeout=cfg.api.report_timeout_s
                )
            except TimeoutError:
                return err(504, "timeout", "The agent took too long; use /v1/report/jobs.")
        observe(res, time.perf_counter() - t0)
        return res

    async def run_job(job_id: str, query: str) -> None:
        job = jobs[job_id]
        async with llm_gate:
            job["status"] = "running"
            t0 = time.perf_counter()
            try:
                res = await run_in_threadpool(svc.report, query)
            except LLMUnavailable:
                job.update(status="error", error="llm_unavailable")
                return
            except Exception as exc:  # noqa: BLE001
                log.error("report job failed: %s", type(exc).__name__)
                job.update(status="error", error="internal_error")
                return
            observe(res, time.perf_counter() - t0)
            job.update(status="done", result=res.model_dump())

    @app.post("/v1/report/jobs", status_code=202, dependencies=[Depends(require_key)])
    async def submit_job(req: ReportRequest) -> dict[str, str]:
        while len(jobs) >= cfg.api.max_jobs:
            jobs.pop(next(iter(jobs)))
        job_id = uuid.uuid4().hex[:16]
        jobs[job_id] = {"status": "queued"}
        task = asyncio.create_task(run_job(job_id, req.query))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return {"job_id": job_id, "status": "queued"}

    @app.get("/v1/report/jobs/{job_id}", dependencies=[Depends(require_key)])
    async def job_status(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"job_id": job_id, **job}

    @app.get("/v1/trace/{trace_id}", dependencies=[Depends(require_key)])
    async def trace(trace_id: str) -> dict[str, Any]:
        t = await run_in_threadpool(svc.trace, trace_id)
        if t is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return t

    @app.get("/metrics")
    async def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def get_app() -> FastAPI:
    """Uvicorn factory: `uvicorn api.main:get_app --factory`."""
    return create_app()
