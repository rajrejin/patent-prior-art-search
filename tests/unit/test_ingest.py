"""Ingest tests. Parsing is tested only against real cached OPS responses."""

from __future__ import annotations

import functools
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from patentrag.config import ROOT, Secrets, load_config
from patentrag.ingest import parse_ops
from patentrag.ingest.corpus import build_cql, select_refs
from patentrag.ingest.ops_client import (
    OpsClient,
    OpsError,
    QuotaExceeded,
    QuotaState,
    week_start,
)

RAW = ROOT / "data" / "raw" / "ops"


@functools.lru_cache(maxsize=1)
def _all_cached() -> tuple[dict, ...]:
    out = []
    for f in sorted(RAW.rglob("*.json.gz")) if RAW.exists() else []:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec["status"] == 200:
            out.append(rec)
    return tuple(out)


def _cached(kind: str) -> list[dict]:
    return [r for r in _all_cached() if r["endpoint"].endswith(kind)]


def _seed_with_everything() -> tuple[dict, dict, dict]:
    """(biblio, claims, family) records of one seed document that has all three cached."""
    by_endpoint = {r["endpoint"]: r for r in _all_cached()}
    for rec in _all_cached():
        if not rec["endpoint"].startswith("/family/publication/docdb/"):
            continue
        ref = rec["endpoint"].rsplit("/", 1)[1]
        base = f"/published-data/publication/docdb/{ref}"
        bib, claims = by_endpoint.get(f"{base}/biblio"), by_endpoint.get(f"{base}/claims")
        if bib and claims:
            return bib, claims, rec
    pytest.skip("no seed with biblio, claims and family cached (run `patentrag fetch`)")


def _need(kind: str) -> dict:
    recs = _cached(kind)
    if not recs:
        pytest.skip(f"no cached real OPS '{kind}' response in data/raw/ops (run `patentrag fetch`)")
    return recs[0]


def test_parse_biblio_real_responses():
    recs = _cached("/biblio")
    if not recs:
        pytest.skip("no cached real OPS 'biblio' response in data/raw/ops (run `patentrag fetch`)")
    parsed = [(r, parse_ops.parse_biblio(r["body"])) for r in recs]
    assert all(p is not None for _, p in parsed)
    for rec, p in parsed:
        assert p is not None
        assert p.pub_id.country and p.pub_id.number
        assert p.pub_date is not None and p.pub_date.year >= 1790
        assert all("/" in c for c in p.cpc)  # some cited documents have no CPC at all
        assert str(p.pub_id) in rec["endpoint"].replace(".", "")
    # OPS does not always supply titles, abstracts, CPC or citations (old JP utility models
    # have none); most documents must have a title.
    assert sum(bool(p.title) for _, p in parsed if p) >= 0.9 * len(parsed)
    assert any(p.cpc for _, p in parsed if p)
    assert any(p.abstract and "abstract" in p.fields_present for _, p in parsed if p)
    assert any(p.cites and p.ipc for _, p in parsed if p)


def test_parse_claims_real_response():
    rec = _need("/claims")
    parsed = parse_ops.parse_claims(rec["body"])
    assert parsed is not None
    lang, text = parsed
    assert "1." in text[:30]
    assert lang in {"en", "de", "fr", ""}


def test_parse_family_real_response():
    _, _, fam = _seed_with_everything()
    members = parse_ops.parse_family(fam["body"])
    assert members
    assert len({str(m) for m in members}) == len(members)


def test_parse_search_real_response():
    recs = [r for r in _cached("/search") if "<ops:publication-reference" in r["body"]]
    if not recs:
        pytest.skip("no cached real search response")
    total, refs = parse_ops.parse_search(recs[0]["body"])
    assert total >= len(refs) > 0
    assert refs[0]["country"] in {"EP", "WO"} and refs[0]["family_id"]


def test_attach_adds_claims_and_family():
    bib, claims, fam = _seed_with_everything()
    p = parse_ops.parse_biblio(bib["body"])
    assert p is not None
    p2 = parse_ops.attach(p, claims["body"], fam["body"])
    assert p2.claims and "claims" in p2.fields_present
    assert "family" in p2.fields_present
    assert str(p2.pub_id) not in {str(m) for m in p2.family_members}


# --- pure logic with hand-computed answers ---------------------------------


def test_week_start_is_monday_gmt():
    # 2026-10-13 is a Tuesday, so the week starts on Monday 2026-10-12.
    assert week_start(datetime(2026, 10, 13, 9, 0, tzinfo=UTC)) == "2026-10-12"
    assert week_start(datetime(2026, 10, 18, 23, 59, tzinfo=UTC)) == "2026-10-12"
    assert week_start(datetime(2026, 10, 19, 0, 0, tzinfo=UTC)) == "2026-10-19"


def test_quota_guard_stops_at_limit_and_persists(tmp_path: Path):
    state = QuotaState(tmp_path / "state.json", weekly_bytes=1000, stop_fraction=0.8)
    state.check()
    state.add(799)
    state.check()  # 799 < 800, still allowed
    state.add(1)
    with pytest.raises(QuotaExceeded):
        state.check()
    reloaded = QuotaState(tmp_path / "state.json", weekly_bytes=1000, stop_fraction=0.8)
    assert reloaded.used == 800
    with pytest.raises(QuotaExceeded):
        reloaded.check()


def test_quota_resets_in_new_week(tmp_path: Path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"week_start": "2020-01-06", "bytes": 10**12, "requests": 5}))
    state = QuotaState(p, weekly_bytes=1000, stop_fraction=0.8)
    assert state.used == 0
    state.check()


def test_build_cql_month_partition():
    cfg = load_config()
    cql = build_cql(cfg, 2020, 2)
    assert 'pd within "20200201 20200229"' in cql
    assert "cpc=H01M10" in cql and "cpc=B60L58" in cql
    assert "pn=EP" in cql and "pn=WO" in cql


def test_select_refs_is_deterministic_and_sorted():
    refs = [{"country": "EP", "number": str(i), "kind": "A1", "family_id": ""} for i in range(50)]
    a = select_refs(refs, 10, seed=0)
    b = select_refs(list(reversed(refs)), 10, seed=0)
    assert a == b and len(a) == 10
    assert a == sorted(a, key=lambda r: f"{r['country']}.{r['number']}.{r['kind']}")
    assert select_refs(refs, 100, seed=0) == sorted(
        refs, key=lambda r: f"{r['country']}.{r['number']}.{r['kind']}"
    )


def _client(tmp_path: Path, handler, sleeps: list[float]) -> OpsClient:
    cfg = load_config()
    cfg.data.raw_dir = str(tmp_path / "raw")
    cfg.data.fetch_state = str(tmp_path / "state.json")
    secrets = Secrets(ops_key="k" * 20, ops_secret="s" * 20)

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("accesstoken"):
            return httpx.Response(200, json={"access_token": "t", "expires_in": "1200"})
        return handler(request)

    return OpsClient(cfg, secrets, sleep=sleeps.append, transport=httpx.MockTransport(route))


def test_client_caches_and_never_refetches(tmp_path: Path):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        return httpx.Response(200, text="<x/>")

    c = _client(tmp_path, handler, [])
    assert c.biblio("EP.1.A1") == "<x/>"
    assert c.biblio("EP.1.A1") == "<x/>"
    assert len(calls) == 1 and c.cache_hits == 1


def test_client_backoff_honours_retry_after(tmp_path: Path):
    seq = iter([httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, text="ok")])
    sleeps: list[float] = []
    c = _client(tmp_path, lambda req: next(seq), sleeps)
    assert c.biblio("EP.2.A1") == "ok"
    assert 7.0 in sleeps


def test_client_404_is_cached_and_tolerated(tmp_path: Path):
    n = {"calls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        n["calls"] += 1
        return httpx.Response(404, text="not found")

    c = _client(tmp_path, handler, [])
    assert c.claims("EP.3.A1") is None
    assert c.claims("EP.3.A1") is None
    assert n["calls"] == 1


def test_client_non_retryable_error_raises(tmp_path: Path):
    c = _client(tmp_path, lambda req: httpx.Response(422, text="bad"), [])
    with pytest.raises(OpsError):
        c.biblio("EP.4.A1")


def test_client_400_on_document_request_is_tolerated_and_cached(tmp_path: Path):
    n = {"calls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        n["calls"] += 1
        return httpx.Response(400, text="invalid reference")

    c = _client(tmp_path, handler, [])
    assert c.biblio("EP.5.A1") is None
    assert c.biblio("EP.5.A1") is None
    assert n["calls"] == 1
