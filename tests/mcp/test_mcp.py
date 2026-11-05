"""MCP server tests with an in-process client: tool listing, schemas, input validation."""

from __future__ import annotations

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from patentrag.mcp_server import MAX_K, build_server


class NeverCalledService:
    """The tests below must be rejected by schema validation before reaching the service."""

    def search(self, *a, **k):
        raise AssertionError("service reached despite invalid input")

    report = search
    trace = search


def run(coro_fn):
    return anyio.run(coro_fn)


def test_tools_are_listed_with_strict_schemas():
    server = build_server(NeverCalledService())  # type: ignore[arg-type]

    async def go():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            return tools

    tools = run(go)
    assert {"search_patents", "run_prior_art_agent", "get_agent_trace"} <= set(tools)
    s = tools["search_patents"].inputSchema
    assert s["required"] == ["query"]
    assert s["properties"]["k"]["maximum"] == MAX_K and s["properties"]["k"]["minimum"] == 1
    assert s["properties"]["query"]["maxLength"] == 4000
    methods = {"auto", "bm25", "vector", "hybrid", "rerank", "graphrag"}
    assert set(s["properties"]["method"]["enum"]) == methods
    assert "provenance" in (tools["search_patents"].description or "")
    r = tools["run_prior_art_agent"].inputSchema
    assert r["required"] == ["idea"] and r["properties"]["idea"]["maxLength"] == 4000
    assert "claim chart" in (tools["run_prior_art_agent"].description or "")


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("search_patents", {"query": "too short"}),
        ("search_patents", {"query": "a valid long enough query", "k": 999}),
        ("search_patents", {"query": "a valid long enough query", "method": "magic"}),
        ("search_patents", {"query": "x" * 4001}),
        ("run_prior_art_agent", {"idea": "short"}),
        ("run_prior_art_agent", {"idea": "x" * 4001}),
        ("get_agent_trace", {"trace_id": "x"}),
    ],
)
def test_invalid_inputs_are_rejected_before_the_service(tool, args):
    server = build_server(NeverCalledService())  # type: ignore[arg-type]

    async def go():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return await client.call_tool(tool, args)

    result = run(go)
    assert result.isError is True
