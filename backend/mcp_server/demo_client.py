"""End-to-end check: launch the server over stdio as a real MCP client and call every tool.

Run from backend/:  python -m mcp_server.demo_client
"""
import asyncio
import json
import sys
import time
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Offline calls need no API key; --live adds the calls that use embeddings / the LLM.
OFFLINE = [
    ("list_documents", {}),
    ("get_citation", {"page": 190, "section": "Credit History"}),
    ("get_citation", {"page": 5}),                      # page outside the corpus
    ("search_guidelines", {"query": "", "top_k": 5}),   # validation error path
    ("analyze_document", {"doc_id": "../../.env"}),     # path traversal attempt
]
LIVE = [
    ("search_guidelines", {"query": "how long after a Chapter 7 bankruptcy discharge can a borrower qualify", "top_k": 2}),
    ("analyze_document", {"doc_id": "risky_application"}),
]
CALL_TIMEOUT = timedelta(seconds=120)


def short(text: str, n: int = 700) -> str:
    return text if len(text) <= n else text[:n] + " ...[truncated]"


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print("TOOLS:", [(t.name, t.annotations.readOnlyHint if t.annotations else None) for t in tools])
            for name, args in OFFLINE + (LIVE if "--live" in sys.argv else []):
                t0 = time.perf_counter()
                res = await session.call_tool(name, args, read_timeout_seconds=CALL_TIMEOUT)
                dt = time.perf_counter() - t0
                body = "".join(c.text for c in res.content if hasattr(c, "text"))
                print(f"\n=== {name} {json.dumps(args)}  isError={res.isError}  {dt:.2f}s")
                print(short(body))


if __name__ == "__main__":
    asyncio.run(main())
