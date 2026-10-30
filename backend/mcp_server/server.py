"""MCP server exposing the underwriting assistant's retrieval and analysis as read-only tools.

Run (stdio transport):  python -m mcp_server.server        (from backend/)
Inspect:                npx @modelcontextprotocol/inspector python -m mcp_server.server
"""
import logging
import os
import sys

import anyio

# Resolve .env and ./chroma_db relative to backend/ no matter where the client launches us.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
# stdout carries the MCP protocol; all logging must go to stderr.
_log_file = os.getenv("MCP_LOG_FILE")  # optional debug log (DEBUG level) for diagnosing client issues
logging.basicConfig(level=logging.DEBUG if _log_file else logging.WARNING,
                    **({"filename": _log_file} if _log_file else {"stream": sys.stderr}))

if _log_file:
    import faulthandler
    faulthandler.dump_traceback_later(25, file=open(_log_file + ".trace", "w"))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

# Import the heavy ML/vector dependencies on the main thread at startup. Loading their native
# extensions lazily from a worker thread hung under client-launched stdio on Windows.
from app.agent import orchestrator  # noqa: E402,F401
from app.services import vector_service  # noqa: E402,F401
from mcp_server import tools  # noqa: E402

mcp = FastMCP("underwriting-assistant")
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)


@mcp.tool(annotations=READ_ONLY)
async def search_guidelines(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search over HUD Handbook 4000.1 borrower-underwriting guidelines.

    Returns up to top_k excerpts (1-10), each with a citation label, page, section and text.
    """
    # blocking embedding + vector search: keep it off the event loop
    return await anyio.to_thread.run_sync(tools.search_guidelines, query, top_k)


@mcp.tool(annotations=READ_ONLY)
def get_citation(page: int, section: str | None = None) -> dict:
    """Return the full text of one handbook page by its printed page number (e.g. 192).

    Optionally filter to a section by name substring. Use after search_guidelines to verify a citation.
    """
    return tools.get_citation(page, section)


@mcp.tool(annotations=READ_ONLY)
def list_documents() -> list[dict]:
    """List the loan applications available to analyze_document (doc_id, filename, size)."""
    return tools.list_documents()


@mcp.tool(annotations=READ_ONLY)
async def analyze_document(doc_id: str) -> dict:
    """Run the cited risk analysis on a listed application (see list_documents).

    Returns score, recommendation and flags, each citing a retrieved handbook page.
    Calls the configured LLM; nothing is stored.
    """
    return await tools.analyze_document(doc_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
