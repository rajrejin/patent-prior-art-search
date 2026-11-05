"""API plumbing: validation, auth, readiness, error schema, metrics. No patent data involved."""

from __future__ import annotations

import pytest
from api.main import create_app
from fastapi.testclient import TestClient

from patentrag.config import load_config
from patentrag.llm.client import LLMUnavailable


class StubService:
    """Stands in for PatentService to test HTTP behaviour only."""

    def __init__(self, ready: dict | None = None, boom: Exception | None = None):
        self._ready = ready or {"indexes": True, "neo4j": True, "llm": True, "scope": "full"}
        self.boom = boom

    def readiness(self):
        return self._ready

    def search(self, *a, **k):
        if self.boom:
            raise self.boom
        raise AssertionError("not reached in these tests")

    def report(self, *a, **k):
        if self.boom:
            raise self.boom
        raise AssertionError("not reached in these tests")

    def trace(self, trace_id):
        return None


@pytest.fixture
def client():
    return TestClient(create_app(StubService(), load_config()), raise_server_exceptions=False)


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert r.headers["x-request-id"]


def test_ready_ok_and_503_when_components_down():
    ok = TestClient(create_app(StubService(), load_config()))
    assert ok.get("/v1/ready").status_code == 200
    down = TestClient(
        create_app(StubService({"indexes": True, "neo4j": False, "llm": False}), load_config())
    )
    r = down.get("/v1/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert any("graph database" in p for p in body["problems"])
    assert any("language model" in p for p in body["problems"])


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "short"},
        {"query": "x" * 5000},
        {"query": "a valid long enough query text", "k": 0},
        {"query": "a valid long enough query text", "k": 51},
        {"query": "a valid long enough query text", "method": "magic"},
    ],
)
def test_search_validation_returns_clean_422(client, payload):
    r = client.post("/v1/search", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_request" and "Traceback" not in r.text


def test_report_validation(client):
    assert client.post("/v1/report", json={"query": "too short"}).status_code == 422
    assert client.post("/v1/report", json={"query": "x" * 4001}).status_code == 422


def test_unhandled_error_hides_stack_trace():
    app = create_app(StubService(boom=RuntimeError("secret internal detail")), load_config())
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/v1/search", json={"query": "a valid long enough query text"})
    assert r.status_code == 500
    assert "secret internal detail" not in r.text and "Traceback" not in r.text
    assert r.json()["error"] == "internal_error" and r.json()["request_id"]


def test_llm_unavailable_maps_to_503():
    app = create_app(StubService(boom=LLMUnavailable("down")), load_config())
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/v1/report", json={"query": "a sufficiently long invention description"})
    assert r.status_code == 503 and r.json()["error"] == "llm_unavailable"


def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("PATENTRAG_API_KEY", "k-test-value")
    c = TestClient(create_app(StubService(), load_config()), raise_server_exceptions=False)
    body = {"query": "a valid long enough query text"}
    assert c.post("/v1/search", json=body).status_code == 401
    assert c.post("/v1/search", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    # the correct key passes auth (the stub then fails, which proves the request got through)
    assert c.post("/v1/search", json=body, headers={"X-API-Key": "k-test-value"}).status_code == 500
    assert c.get("/v1/health").status_code == 200  # health stays open


def test_trace_not_found_and_metrics(client):
    assert client.get("/v1/trace/abcd1234").status_code == 404
    m = client.get("/metrics")
    assert m.status_code == 200 and "patentrag_requests_total" in m.text


def test_cors_off_by_default(client):
    r = client.get("/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


class ReportingService(StubService):
    """Returns a minimal agent response, or raises, for the report-job tests."""

    def __init__(self, fail: bool = False):
        super().__init__()
        self.fail = fail

    def report(self, *a, **k):
        from patentrag.schemas import ReportResponse

        if self.fail:
            raise RuntimeError("agent crashed")
        return ReportResponse(
            trace_id="t1",
            summary="s",
            items=[],
            limitations="l",
            claim_support=[],
            grounded=False,
            attempts=1,
            warning="no rows",
            took_ms=1.0,
        )


def poll(c, job_id):
    for _ in range(50):
        body = c.get(f"/v1/report/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
    raise AssertionError("job did not finish")


def test_report_job_runs_in_background_and_returns_the_result():
    with TestClient(create_app(ReportingService(), load_config())) as c:
        r = c.post("/v1/report/jobs", json={"query": "a sufficiently long invention description"})
        assert r.status_code == 202 and r.json()["status"] == "queued"
        body = poll(c, r.json()["job_id"])
        assert body["status"] == "done" and body["result"]["trace_id"] == "t1"
        assert c.get("/v1/report/jobs/unknown").status_code == 404


def test_report_job_failure_is_reported_without_details():
    with TestClient(create_app(ReportingService(fail=True), load_config())) as c:
        r = c.post("/v1/report/jobs", json={"query": "a sufficiently long invention description"})
        body = poll(c, r.json()["job_id"])
        assert body == {"job_id": r.json()["job_id"], "status": "error", "error": "internal_error"}
        bad = c.post("/v1/report/jobs", json={"query": "short"})
        assert bad.status_code == 422
