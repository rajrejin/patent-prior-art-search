"""Single entry point `patentrag`: data, indexes, evaluation, agent and servers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from patentrag.config import ROOT, load_config
from patentrag.environment import write_environment
from patentrag.logging_utils import setup_logging

app = typer.Typer(help="Agentic RAG for European patent prior-art search.", no_args_is_help=True)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def env() -> None:
    """Detect hardware and write results/environment.json."""
    info = write_environment(ROOT / "results" / "environment.json")
    typer.echo(f"wrote results/environment.json (ram_gb={info['ram_gb']})")


@app.command()
def config() -> None:
    """Print the config hash."""
    typer.echo(load_config().config_hash())


@app.command("check-credentials")
def check_credentials(live: bool = typer.Option(True, help="Also call the OPS API.")) -> None:
    """Silently verify the OPS credentials in .env (never prints values)."""
    from patentrag.ingest.credentials import check_env_shape, check_live

    results = check_env_shape()
    if all(r.ok for r in results) and live:
        results += check_live()
    for r in results:
        typer.echo(f"{'PASS' if r.ok else 'FAIL'}  {r.name}  {r.reason}".rstrip())
    if not all(r.ok for r in results):
        raise typer.Exit(code=1)


@app.command()
def fetch(scope: str = typer.Option("full", help="slice (one month) or full")) -> None:
    """Fetch patents from EPO OPS (resumable, quota guarded)."""
    from patentrag.ingest.corpus import fetch_documents, make_closure, make_selection
    from patentrag.ingest.ops_client import OpsClient, QuotaExceeded

    cfg = load_config()
    client = OpsClient(cfg)
    try:
        seeds = make_selection(client, cfg, scope)
        typer.echo(f"seeds: {len(seeds)} publications")
        res = fetch_documents(client, cfg, seeds)
        if not res["quota_stopped"]:
            closure = make_closure(client, cfg, scope, seeds)
            typer.echo(f"citation closure: {len(closure)} additional documents")
            res = fetch_documents(client, cfg, seeds, closure)
    except QuotaExceeded as exc:
        typer.echo(f"quota guard: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    finally:
        client.close()
    typer.echo(
        f"fetched {res['done']}/{res['total']} (network requests: {client.network_requests})"
    )
    if res["quota_stopped"]:
        typer.echo("stopped by quota guard; rerun `patentrag fetch` later to resume.", err=True)
        raise typer.Exit(code=3)


@app.command("build-corpus")
def build_corpus(scope: str = typer.Option("full")) -> None:
    """Parse cached raw responses into the normalised corpus and write corpus stats."""
    import json

    from patentrag.ingest.corpus import (
        make_closure,
        make_selection,
        parse_selection,
        write_corpus,
    )
    from patentrag.ingest.ops_client import OpsClient
    from patentrag.ingest.stats import corpus_stats

    cfg = load_config()
    client = OpsClient(cfg)
    try:
        seeds = make_selection(client, cfg, scope)
        closure = make_closure(client, cfg, scope, seeds)
        patents, dropped = parse_selection(client, cfg, seeds, closure)
    finally:
        client.close()
    manifest = write_corpus(cfg, scope, patents)
    manifest["dropped"] = dropped
    stats = corpus_stats(patents, cfg.evaluation.min_relevant)
    stats["manifest"] = manifest
    out = ROOT / "results" / f"corpus_stats_{scope}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    typer.echo(f"corpus: {len(patents)} patents -> {out.name}")


@app.command("build-graph")
def build_graph(
    scope: str = typer.Option("full"),
    reset: bool = typer.Option(False, help="Delete all nodes first (this project's database)."),
) -> None:
    """Load the corpus into Neo4j and verify the counts against the corpus."""
    import json

    from patentrag.graph.loader import connect, load_corpus_into_graph
    from patentrag.ingest.corpus import load_corpus

    cfg = load_config()
    patents = load_corpus(cfg, scope)
    driver = connect()
    try:
        if reset:
            with driver.session() as s:
                s.run("MATCH (n) DETACH DELETE n").consume()
        res = load_corpus_into_graph(driver, patents)
    finally:
        driver.close()
    out = ROOT / "results" / f"graph_load_{scope}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    typer.echo(f"graph counts match corpus: {res['match']} -> {out.name}")
    if not res["match"]:
        raise typer.Exit(code=1)


@app.command("build-index")
def build_index(scope: str = typer.Option("full")) -> None:
    """Build the vector and BM25 indexes."""
    from patentrag.index.build import build_indexes

    m = build_indexes(load_config(), scope)
    typer.echo(f"indexed {m['n_patents']} patents, {m['n_chunks']} chunks ({m['seconds']})")


def _stack(scope: str) -> Any:
    from patentrag.stack import Stack

    return Stack(load_config(), scope)


def _agent(stack: Any, *, nli: Any = None, threshold: float | None = None) -> Any:
    from patentrag.agent.loop import PriorArtAgent
    from patentrag.evaluation.params import load_tuned_agent
    from patentrag.llm.ollama_backend import OllamaClient

    cfg = stack.cfg
    return PriorArtAgent(
        cfg,
        stack,
        OllamaClient(cfg),
        params=load_tuned_agent(cfg, stack.scope),
        nli_fn=nli.entailment if nli is not None else None,
        nli_threshold=threshold,
    )


@app.command()
def evaluate(
    scope: str = typer.Option("full"),
    split: str = typer.Option("val", help="val or test (test is touched for final numbers only)"),
    methods: str = typer.Option("bm25,vector,hybrid,rerank,graphrag,agent"),
    name: str = typer.Option("", help="output name under results/"),
    default_params: bool = typer.Option(False, help="Untuned GraphRAG parameters (config)."),
) -> None:
    """Run the retrieval evaluation and print the comparison table."""
    from patentrag.evaluation.harness import ensure_testset, evaluate_methods, summary_table
    from patentrag.evaluation.params import load_tuned_rerank

    stack = _stack(scope)
    cfg = stack.cfg
    retrievers: dict[str, Any] = {}
    extra_runs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for m in [m.strip() for m in methods.split(",") if m.strip()]:
        if m == "bm25":
            retrievers[m] = stack.bm25()
        elif m == "vector":
            retrievers[m] = stack.vector
        elif m == "hybrid":
            retrievers[m] = stack.hybrid()
        elif m == "rerank":
            retrievers[m] = stack.rerank(load_tuned_rerank(cfg, scope))
        elif m in ("graphrag", "graphrag_leaky"):
            # graphrag_leaky switches off leakage control 2 (labelled demonstration only)
            retrievers[m] = stack.graphrag(cfg.graph if default_params else None)
        elif m == "agent":
            from patentrag.evaluation.agent_eval import agent_run_result

            val, test, _ = ensure_testset(cfg, scope, list(stack.patents.values()))
            queries = val if split == "val" else test
            run, beh = agent_run_result(cfg, scope, _agent(stack), queries, split)
            extra_runs[m] = run
            extra["agent_behaviour"] = beh
        else:
            raise typer.BadParameter(f"unknown method {m}")
    res = evaluate_methods(
        cfg,
        scope,
        retrievers,
        split,
        name or f"eval_{scope}_{split}",
        hide_outgoing={"graphrag_leaky": False},
        extra_runs=extra_runs,
        extra=extra,
    )
    typer.echo(f"queries: {res['n_queries']}")
    typer.echo(summary_table(res, cfg.evaluation.ks))


@app.command()
def tune(scope: str = typer.Option("full")) -> None:
    """Tune hybrid weight, graph signals and re-ranker on validation queries only."""
    from patentrag.evaluation.tuning import tune as run_tune

    typer.echo(json.dumps(run_tune(load_config(), scope, _stack(scope)), indent=2))


@app.command("tune-agent")
def tune_agent(scope: str = typer.Option("full")) -> None:
    """Run the retrieval agent on validation queries and choose its feature-search weight."""
    from patentrag.evaluation.agent_eval import tune_agent as run

    stack = _stack(scope)
    typer.echo(json.dumps(run(stack.cfg, scope, _agent(stack)), indent=2))


@app.command()
def ablate(scope: str = typer.Option("full"), split: str = typer.Option("val")) -> None:
    """Run the ablation study with the tuned parameters."""
    from patentrag.evaluation.ablations import run_ablations

    out = run_ablations(load_config(), scope, _stack(scope), split)
    typer.echo(f"ablations on {out['n_queries']} {split} queries -> results/")


@app.command()
def robustness(scope: str = typer.Option("full")) -> None:
    """Query-form, language and corpus-size study on the test queries."""
    from patentrag.evaluation.params import load_tuned_rerank
    from patentrag.evaluation.robustness import run_robustness

    stack = _stack(scope)
    retrievers: dict[str, Any] = {
        "bm25": stack.bm25(),
        "vector": stack.vector,
        "hybrid": stack.hybrid(),
    }
    rr = load_tuned_rerank(stack.cfg, scope)
    if rr.enabled:
        retrievers["rerank"] = stack.rerank(rr)
    out = run_robustness(stack.cfg, scope, retrievers)
    typer.echo(f"robustness written for {out['n_test_queries']} test queries")


@app.command()
def faithfulness(
    scope: str = typer.Option("full"),
    n_test: int = typer.Option(100, help="Test queries to run the full agent on"),
    n_cal: int = typer.Option(12, help="Validation queries for the NLI threshold"),
) -> None:
    """Run the full agent (reports) and evaluate grounding; resumable checkpoints.

    The NLI threshold is calibrated on validation reports before the test reports run.
    """
    from patentrag.evaluation import faith_eval
    from patentrag.evaluation.agent_eval import behaviour, report_queries, run_queries, runs_path
    from patentrag.faithfulness.nli import NliJudge
    from patentrag.faithfulness.threshold import load_threshold

    stack = _stack(scope)
    cfg = stack.cfg
    nli = NliJudge(cfg.faithfulness.nli_model)
    qs = report_queries(cfg, scope, n_cal, n_test)
    agent = _agent(stack, nli=nli, threshold=0.5)
    val = run_queries(agent, qs["val"], runs_path("report", "val", scope), retrieval_only=False)
    typer.echo(f"threshold: {faith_eval.calibrate_threshold(nli, val)}")
    th = load_threshold()
    agent = _agent(stack, nli=nli, threshold=th)
    recs = run_queries(agent, qs["test"], runs_path("report", "test", scope), retrieval_only=False)
    out = faith_eval.summarise(recs, th)
    out["negative_control"] = faith_eval.negative_control(agent.llm, nli, recs, th)
    out["agent_behaviour"] = behaviour(recs)
    out["n_test_queries"] = len(recs)
    out["model"] = cfg.llm.model
    out["nli_model"] = cfg.faithfulness.nli_model
    (ROOT / "results" / f"faithfulness_{scope}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    typer.echo(json.dumps(out, indent=2))


@app.command("faith-audit")
def faith_audit(
    scope: str = typer.Option("full"),
    model: str = typer.Option("qwen2.5:7b-instruct", help="Judge model not used by the pipeline"),
) -> None:
    """Re-judge the claim-chart rows of the test reports with an independent, larger model."""
    from patentrag.evaluation import faith_eval
    from patentrag.evaluation.agent_eval import load_records, runs_path
    from patentrag.llm.ollama_backend import OllamaClient

    cfg = load_config()
    if model == cfg.llm.model:
        raise typer.BadParameter("the audit judge must differ from the pipeline model")
    cfg.llm.model = model
    recs = list(load_records(runs_path("report", "test", scope)).values())
    out = faith_eval.audit(OllamaClient(cfg), recs)
    dest = ROOT / "results" / f"faithfulness_audit_{scope}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    typer.echo(json.dumps(out, indent=2))


@app.command("serve-mcp")
def serve_mcp(transport: str = typer.Option("stdio", help="stdio or streamable-http")) -> None:
    """Serve the MCP tools (stdio for Claude Desktop)."""
    from patentrag.mcp_server import main as run

    run(transport)


@app.command("serve-api")
def serve_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the FastAPI app."""
    import uvicorn

    uvicorn.run("api.main:get_app", factory=True, host=host, port=port)


@app.command()
def report(
    query: str = typer.Argument(..., help="Invention idea, or @file.txt to read it from a file"),
) -> None:
    """Run the prior-art agent and print its verified claim chart and steps as JSON."""
    from patentrag.service import PatentService

    text = Path(query[1:]).read_text(encoding="utf-8") if query.startswith("@") else query
    typer.echo(PatentService().report(text).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
