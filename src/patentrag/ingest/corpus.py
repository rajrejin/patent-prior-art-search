"""Search partitioning, document selection, resumable fetching and corpus building."""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from patentrag.config import Config
from patentrag.ingest import parse_ops
from patentrag.ingest.ops_client import OpsClient, QuotaExceeded
from patentrag.models import Patent

log = logging.getLogger(__name__)

PAGE = 100


def month_ranges(year_start: int, year_end: int) -> Iterator[tuple[int, int]]:
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):
            yield y, m


def build_cql(cfg: Config, year: int, month: int) -> str:
    cpcs = " or ".join(f"cpc={p.rstrip('/')}" for p in cfg.data.cpc_prefixes)
    auth = " or ".join(f"pn={a}" for a in cfg.data.authorities)
    last = calendar.monthrange(year, month)[1]
    return f'({cpcs}) and ({auth}) and pd within "{year}{month:02d}01 {year}{month:02d}{last:02d}"'


def ref_string(ref: dict[str, str]) -> str:
    return f"{ref['country']}.{ref['number']}.{ref['kind']}"


def collect_month(client: OpsClient, cfg: Config, year: int, month: int) -> list[dict[str, str]]:
    """All publication references for one month (paged, cached)."""
    cql = build_cql(cfg, year, month)
    refs: list[dict[str, str]] = []
    start = 1
    total = None
    while total is None or start <= total:
        end = start + PAGE - 1
        t, page = parse_ops.parse_search(client.search(cql, start, end))
        total = t
        if total > 2000:
            raise RuntimeError(f"{year}-{month:02d} has {total} hits (> 2000 cap); partition more")
        refs.extend(page)
        if not page:
            break
        start = end + 1
    return refs


def select_refs(refs: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    """Deterministic seeded sample, returned in a stable order."""
    ordered = sorted(refs, key=ref_string)
    if len(ordered) <= n:
        return ordered
    rng = random.Random(seed)  # noqa: S311
    return sorted(rng.sample(ordered, n), key=ref_string)


def selection_path(cfg: Config, scope: str) -> Path:
    return cfg.path(cfg.data.corpus_dir) / f"selection_{scope}.json"


def make_selection(client: OpsClient, cfg: Config, scope: str) -> list[dict[str, str]]:
    path = selection_path(cfg, scope)
    if path.exists():
        data: list[dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
        return data
    if scope == "slice":
        months = [(cfg.data.slice_year, cfg.data.slice_month)]
        per_month = cfg.data.slice_size
    else:
        months = list(month_ranges(cfg.data.year_start, cfg.data.year_end))
        per_month = cfg.data.seeds_per_month
    chosen: list[dict[str, str]] = []
    for y, m in months:
        refs = collect_month(client, cfg, y, m)
        picked = select_refs(refs, per_month, cfg.seed)
        log.info("month %d-%02d: %d hits, selected %d", y, m, len(refs), len(picked))
        chosen.extend(picked)
    chosen = sorted(chosen, key=ref_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    return chosen


def closure_path(cfg: Config, scope: str) -> Path:
    return cfg.path(cfg.data.corpus_dir) / f"closure_{scope}.json"


def seed_biblio(client: OpsClient, ref: dict[str, str]) -> Patent | None:
    body = _cached_body(client, f"/published-data/publication/docdb/{ref_string(ref)}/biblio")
    return parse_ops.parse_biblio(body) if body else None


def make_closure(client: OpsClient, cfg: Config, scope: str, seeds: list[dict[str, str]]) -> list:
    """Documents cited by the seeds (any authority) that are not themselves seeds.

    Computed from cached seed biblio only. One reference per document key, in a stable order.
    """
    path = closure_path(cfg, scope)
    seed_keys = {f"{r['country']}{r['number']}" for r in seeds}
    found: dict[str, dict[str, str]] = {}
    for ref in seeds:
        p = seed_biblio(client, ref)
        if p is None:
            continue
        for c in p.cites:
            if c.doc_key in seed_keys or c.doc_key in found:
                continue
            found[c.doc_key] = {"country": c.country, "number": c.number, "kind": c.kind}
    refs = [found[k] for k in sorted(found)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(refs, indent=1), encoding="utf-8")
    return refs


def fetch_documents(
    client: OpsClient,
    cfg: Config,
    seeds: list[dict[str, str]],
    closure: list[dict[str, str]] | None = None,
) -> dict[str, int]:
    """Fetch seeds (biblio, claims, family) then closure documents (biblio, claims).

    Cached responses are free, so a rerun resumes where the previous run stopped.
    """
    jobs = [(r, True) for r in seeds] + [(r, False) for r in (closure or [])]
    done = 0
    try:
        for i, (ref, is_seed) in enumerate(jobs, 1):
            s = ref_string(ref)
            client.biblio(s)
            if cfg.data.fetch_claims:
                client.claims(s)
            if is_seed:
                client.family(s)
            if cfg.data.fetch_descriptions:
                client.description(s)
            done = i
            if i % 25 == 0:
                log.info(
                    "fetched %d/%d (network=%d cache=%d, week bytes=%d)",
                    i,
                    len(jobs),
                    client.network_requests,
                    client.cache_hits,
                    client.quota.used,
                )
    except QuotaExceeded as exc:
        log.warning("quota guard stopped fetching at %d/%d: %s", done, len(jobs), exc)
        return {"done": done, "total": len(jobs), "quota_stopped": 1}
    return {"done": done, "total": len(jobs), "quota_stopped": 0}


def _cached_body(client: OpsClient, endpoint: str) -> str | None:
    rec = client.cached(endpoint)
    if rec is None or rec["status"] != 200:
        return None
    return str(rec["body"])


def has_text(p: Patent) -> bool:
    return bool(p.abstract or p.claims)


def parse_selection(
    client: OpsClient,
    cfg: Config,
    seeds: list[dict[str, str]],
    closure: list[dict[str, str]] | None = None,
) -> tuple[list[Patent], dict[str, int]]:
    """Parse cached raw responses only (no network). Returns (patents, dropped counts)."""
    patents: dict[str, Patent] = {}
    dropped = {"no_biblio": 0, "no_text": 0}
    for origin, refs in (("seed", seeds), ("cited", closure or [])):
        for ref in refs:
            s = ref_string(ref)
            base = f"/published-data/publication/docdb/{s}"
            bib = _cached_body(client, f"{base}/biblio")
            p = parse_ops.parse_biblio(bib) if bib else None
            if p is None:
                dropped["no_biblio"] += 1
                continue
            p = parse_ops.attach(
                p,
                _cached_body(client, f"{base}/claims"),
                _cached_body(client, f"/family/publication/docdb/{s}")
                if origin == "seed"
                else None,
            )
            p = p.model_copy(update={"origin": origin})
            if origin == "cited" and not has_text(p):
                dropped["no_text"] += 1
                continue
            patents.setdefault(str(p.pub_id), p)  # dedupe by publication id; seeds win
    return [patents[k] for k in sorted(patents)], dropped


def corpus_paths(cfg: Config, scope: str) -> tuple[Path, Path]:
    d = cfg.path(cfg.data.corpus_dir)
    return d / f"patents_{scope}.jsonl", d / f"manifest_{scope}.json"


def write_corpus(cfg: Config, scope: str, patents: list[Patent]) -> dict[str, Any]:
    out, man = corpus_paths(cfg, scope)
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for p in patents:
            line = p.model_dump_json()
            h.update(line.encode("utf-8"))
            fh.write(line + "\n")
    manifest = {
        "scope": scope,
        "n_patents": len(patents),
        "corpus_hash": h.hexdigest()[:16],
        "config_hash": cfg.config_hash(),
    }
    man.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_corpus(cfg: Config, scope: str) -> list[Patent]:
    out, _ = corpus_paths(cfg, scope)
    with out.open(encoding="utf-8") as fh:
        return [Patent.model_validate_json(line) for line in fh if line.strip()]
