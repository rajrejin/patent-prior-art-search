"""MCP server (FastMCP) exposing prior-art search and the prior-art agent.

Read-only with respect to the corpus: the only thing written is the local SQLite session cache
and trace store. Run over stdio (default) or streamable HTTP.
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from patentrag.schemas import Method, ReportResponse, SearchResponse
from patentrag.service import PatentService

MAX_QUERY = 4000
MAX_K = 20

INSTRUCTIONS = (
    "Prior-art search over a corpus of European (EP/WO) battery patents and the documents they "
    "cite. Use search_patents to find related patents quickly. Use run_prior_art_agent for a "
    "verified claim chart: a local LLM agent searches the idea and its features, inspects "
    "candidates, maps features to evidence sentences and checks every row against the patent "
    "text. It is slower (local LLM). Results describe text similarity, not legal novelty or "
    "freedom to operate."
)


def build_server(service: PatentService | None = None) -> FastMCP:
    svc = service or PatentService()
    mcp = FastMCP("patent-prior-art", instructions=INSTRUCTIONS)

    @mcp.tool()
    def search_patents(
        query: Annotated[
            str,
            Field(
                min_length=10,
                max_length=MAX_QUERY,
                description="Invention idea or technical description in plain language "
                "(a title plus abstract works best). Do not include patent numbers.",
            ),
        ],
        k: Annotated[int, Field(ge=1, le=MAX_K, description="Number of results")] = 10,
        method: Annotated[
            Method,
            Field(
                description="auto (default): the configuration chosen on validation queries "
                "(hybrid, plus cross-encoder re-ranking if it helped). hybrid: BM25 and "
                "embedding scores combined. rerank: hybrid plus cross-encoder. graphrag: hybrid "
                "plus knowledge-graph signals, with provenance. vector, bm25: single methods."
            ),
        ] = "auto",
    ) -> SearchResponse:
        """Find patents related to a described invention.

        Returns ranked patents with title, date, applicants, CPC codes, a text snippet and
        `provenance` explaining why each was retrieved (for example a shared CPC group, a
        citation link to a similar patent, or vector similarity only). If the graph database
        is unavailable a graphrag call falls back to hybrid search and sets `degraded`.
        """
        return svc.search(query, k=k, method=method)

    @mcp.tool()
    def run_prior_art_agent(
        idea: Annotated[
            str,
            Field(
                min_length=20,
                max_length=MAX_QUERY,
                description="Description of the invention idea to assess against prior art.",
            ),
        ],
    ) -> ReportResponse:
        """Run the prior-art agent and return a verified claim chart.

        The agent chooses its own steps: search the whole idea, split it into features, search
        features, read candidates, look up knowledge-graph neighbours, then build a claim chart
        that maps each feature to an evidence sentence of a retrieved patent (sentences ranked
        by embedding similarity). Every row is checked: two judges must find that the sentence
        states the feature; a rejected row is retried with the next sentence, and pairs with no
        accepted sentence are left out and counted in `removed_rows`.
        `agent_steps` lists every decision. Takes tens of seconds on a GPU, minutes on CPU.
        """
        return svc.report(idea)

    @mcp.tool()
    def get_agent_trace(
        trace_id: Annotated[str, Field(min_length=4, max_length=64)],
    ) -> dict[str, Any]:
        """Return the ordered agent steps, timings and report attempts of an earlier run."""
        t = svc.trace(trace_id)
        if t is None:
            raise ValueError(f"no trace with id {trace_id}")
        return t

    return mcp


def main(transport: str = "stdio") -> None:
    build_server().run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
