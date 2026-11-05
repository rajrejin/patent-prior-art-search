"""Streamlit UI. HTTP client only: every call goes to the FastAPI service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("PATENTRAG_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("PATENTRAG_API_KEY", "")
RESULTS = Path(os.environ.get("PATENTRAG_RESULTS", Path(__file__).resolve().parents[1] / "results"))

st.set_page_config(page_title="Patent prior-art search", layout="wide")


def call(method: str, path: str, **kw: Any) -> tuple[int, Any]:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        r = httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=900, **kw)
        return r.status_code, r.json()
    except (httpx.HTTPError, ValueError) as exc:
        return 0, {"error": "unreachable", "detail": type(exc).__name__}


def provenance_graph(query: str, hits: list[dict[str, Any]]) -> str:
    """HTML for a pyvis subgraph built from each result's provenance strings."""
    from pyvis.network import Network

    net = Network(height="520px", width="100%", directed=True, bgcolor="#ffffff")
    net.add_node("QUERY", label="query", color="#d62728", shape="star", size=26)
    ids = {h["pub_id"] for h in hits}
    for h in hits:
        net.add_node(h["pub_id"], label=h["pub_id"], title=h["title"], color="#1f77b4", size=16)
    for h in hits:
        pid = h["pub_id"]
        for why in h["provenance"]:
            parts = why.split(":")
            kind = parts[0]
            if kind in ("vector", "vector-seed", "bm25", "hybrid", "hybrid-seed"):
                net.add_edge("QUERY", pid, label=kind.split("-")[0], color="#7f7f7f")
            elif kind == "cpc":
                node = f"CPC {parts[1]}"
                net.add_node(node, label=parts[1], color="#2ca02c", shape="box", size=12)
                net.add_edge(pid, node, label="classified", color="#2ca02c")
            elif kind == "citation" and len(parts) >= 3:
                via = parts[2]
                if via not in ids:
                    net.add_node(via, label=via, color="#aec7e8", size=10, title="seed (not shown)")
                src, dst = (pid, via) if parts[1] == "cites" else (via, pid)
                net.add_edge(src, dst, label="cites", color="#ff7f0e")
            elif kind == "family" and len(parts) >= 2:
                via = parts[1]
                if via not in ids:
                    net.add_node(via, label=via, color="#aec7e8", size=10)
                net.add_edge(pid, via, label="family", color="#9467bd", dashes=True)
    return str(net.generate_html(notebook=False))


def show_hits(resp: dict[str, Any]) -> None:
    if resp.get("degraded"):
        st.warning(f"Graph unavailable, showing hybrid results. {resp.get('degraded_reason', '')}")
    cols = st.columns(3)
    cols[0].metric("Method", resp["method"])
    cols[1].metric("Time (ms)", resp["took_ms"])
    cols[2].metric("Candidates", resp.get("n_candidates") or "-")
    if resp.get("inferred_cpc"):
        st.caption("Inferred CPC groups: " + ", ".join(resp["inferred_cpc"]))
    for h in resp["hits"]:
        with st.expander(f"{h['rank']}. {h['pub_id']}  {h['title']}  (score {h['score']})"):
            st.write(h["snippet"])
            st.caption(
                f"{h['pub_date']} | {', '.join(h['applicants'])} | CPC {', '.join(h['cpc'])}"
            )
            st.markdown("**Why retrieved:** " + (", ".join(h["provenance"]) or "n/a"))
    if resp["method"] == "graphrag" and resp["hits"]:
        import streamlit.components.v1 as components

        st.subheader("Subgraph")
        components.html(provenance_graph("", resp["hits"]), height=540, scrolling=True)


def run_agent_job(query: str, poll_s: float = 3.0, max_wait_s: float = 3600.0) -> tuple[int, Any]:
    """Submit an agent run as a background job and poll until it finishes."""
    import time

    code, body = call("POST", "/v1/report/jobs", json={"query": query})
    if code != 202:
        return code, body
    job_id = body["job_id"]
    waited = 0.0
    while waited < max_wait_s:
        time.sleep(poll_s)
        waited += poll_s
        code, body = call("GET", f"/v1/report/jobs/{job_id}")
        if code != 200:
            return code, body
        if body["status"] == "done":
            return 200, body["result"]
        if body["status"] == "error":
            return 500, {"error": body.get("error", "error"), "detail": "the agent run failed"}
    return 504, {"error": "timeout", "detail": f"no result after {max_wait_s:.0f} s"}


def show_agent(body: dict[str, Any]) -> None:
    st.subheader("Agent steps")
    for step in body["agent_steps"]:
        who = {"planner": "LLM decided", "fallback": "fallback", "guardrail": "guardrail"}
        label = f"{step['n']}. {step['tool']}  [{who.get(step['source'], step['source'])}]"
        with st.expander(label):
            if step["thought"]:
                st.markdown(f"*{step['thought']}*")
            if step["args"]:
                st.code(json.dumps(step["args"]), language="json")
            if step["error"]:
                st.caption(f"replaced because: {step['error']}")
            st.write(step["observation"])
            st.caption(f"{step['seconds']} s")


def show_chart(body: dict[str, Any]) -> None:
    if body["warning"]:
        st.warning(body["warning"])
    elif body["grounded"]:
        st.success("Every row below passed the grounding checks.")
    st.subheader("Invention features")
    for i, f in enumerate(body["features"], 1):
        st.markdown(f"**F{i}** {f}")
    st.subheader("Claim chart")
    if not body["items"]:
        st.info("No verified prior art for this idea.")
    for it in body["items"]:
        st.markdown(f"**{it['pub_id']}**  {it['title']}")
        st.table(
            [
                {"feature": r["feature"], "evidence in the patent": r["evidence"]}
                for r in it["chart"]
            ]
        )
    st.caption(
        f"Feature and patent pairs without verified evidence: {body['removed_rows']}. Tries: "
        f"{body['attempts']}. {body['limitations']}"
    )
    with st.expander("All statement checks (including rejected rows)"):
        for c in body["claim_support"]:
            mark = "supported" if c["supported"] else "NOT SUPPORTED"
            st.markdown(
                f"- **{mark}** ({c['source']}) {c['claim']}  "
                f"`llm={c['llm_supported']} nli={c['nli_probability']}`"
            )


tab_search, tab_agent, tab_trace, tab_results = st.tabs(
    ["Search", "Prior-art agent", "Agent trace", "Results"]
)

with tab_search:
    q = st.text_area("Describe the invention", height=140, key="q_search")
    c1, c2 = st.columns(2)
    method = c1.selectbox("Method", ["auto", "hybrid", "rerank", "graphrag", "vector", "bm25"])
    k = c2.slider("Results", 1, 20, 10)
    if st.button("Search", disabled=len(q.strip()) < 10):
        code, body = call("POST", "/v1/search", json={"query": q, "k": k, "method": method})
        if code == 200:
            show_hits(body)
        else:
            st.error(f"{body.get('error')}: {body.get('detail')}")

with tab_agent:
    q2 = st.text_area("Invention idea", height=140, key="q_report")
    st.caption(
        "The agent runs a local language model: about a minute on a GPU, several minutes on a "
        "laptop CPU. The run continues in the background while this page waits."
    )
    if st.button("Run agent", disabled=len(q2.strip()) < 20):
        with st.spinner("The agent is searching, reading and verifying..."):
            code, body = run_agent_job(q2)
        if code != 200:
            st.error(f"{body.get('error')}: {body.get('detail')}")
        else:
            st.session_state["last_trace"] = body["trace_id"]
            left, right = st.columns([3, 2])
            with left:
                show_chart(body)
            with right:
                show_agent(body)
                st.caption("Agent ranking: " + ", ".join(body["ranking"]))
            st.caption(f"Trace id: {body['trace_id']}")

with tab_trace:
    tid = st.text_input("Trace id", value=st.session_state.get("last_trace", ""))
    if tid and st.button("Load trace"):
        code, body = call("GET", f"/v1/trace/{tid}")
        if code == 200:
            for step in body["steps"]:
                st.markdown(
                    f"**{step.get('n', '')} {step['tool']}**  ({step.get('seconds', '')} s)"
                )
                st.json(step, expanded=False)
        else:
            st.error(f"{body.get('error')}: {body.get('detail')}")

with tab_results:
    st.write("Numbers below are read from files in `results/`, produced by `patentrag evaluate`.")
    files = sorted(RESULTS.glob("eval_*.json")) if RESULTS.exists() else []
    if not files:
        st.info("No evaluation results yet. Run `patentrag evaluate`.")
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        st.subheader(f.stem)
        rows = []
        for name, s in data["summary"].items():
            rows.append(
                {
                    "method": name,
                    **{
                        m: f"{s[m]['mean']:.3f} [{s[m]['low']:.3f}, {s[m]['high']:.3f}]"
                        for m in ("ndcg@10", "mrr", "r@10", "p@10")
                        if m in s
                    },
                }
            )
        st.table(rows)
        st.caption(f"{data['n_queries']} queries, split={data['split']}, scope={data['scope']}")
