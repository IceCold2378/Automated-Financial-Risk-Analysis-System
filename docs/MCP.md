# MCP server

Exposes the assistant's retrieval and analysis as four **read-only** tools for any MCP client (Claude Desktop, IDE agents, the MCP Inspector). Implemented with the official `mcp` SDK (`FastMCP`, pinned `<2`: v2 renamed the API), stdio transport, in `backend/mcp_server/`. The tools reuse the existing service layer (`vector_service`, `agent.orchestrator`); nothing is duplicated.

| Tool | Arguments | Returns | Uses LLM/embeddings |
|---|---|---|---|
| `search_guidelines` | `query` (1-500 chars), `top_k` (1-10, default 5) | excerpts with `citation`, `page`, `section`, `text` | embeddings |
| `get_citation` | `page` (int), `section` (optional name substring) | full text of that handbook page + citation label | no |
| `list_documents` | none | available applications (`doc_id`, `filename`, `characters`) | no |
| `analyze_document` | `doc_id` (from `list_documents`) | score, level, recommendation, flags each with a handbook citation, `grounded_flags` | embeddings + LLM |

**Safety:** every tool is annotated `readOnlyHint`. Nothing writes to the database, vector store or disk. `doc_id` must match `^[a-z0-9][a-z0-9_-]{0,63}$` *and* be listed in the documents folder (`backend/data/test_samples`, override with `MCP_DOCS_DIR`), so path traversal is rejected. Inputs are validated and errors come back as MCP tool errors (`isError=true`).

## Connect
Run from `backend/` (the server `chdir`s there itself, so `.env` and `chroma_db` resolve from any client):
```bash
python -m mcp_server.server                      # stdio
npx @modelcontextprotocol/inspector python -m mcp_server.server
```
Claude Desktop (`claude_desktop_config.json`):
```json
{ "mcpServers": { "underwriting-assistant": {
    "command": "<repo>/venv/Scripts/python.exe",
    "args": ["-m", "mcp_server.server"],
    "cwd": "<repo>/backend" } } }
```
Needs `OPENROUTER_API_KEY` in `backend/.env` for the two tools that call the API. The first `search_guidelines` call builds the vector index (about 15 s); later calls reuse it.

## Verified calls (real MCP client over stdio)
`python -m mcp_server.demo_client` launches the server as a subprocess and calls the tools through the MCP SDK's `ClientSession`. Captured 2026-10-15; `tools/list` returned all four tools with `readOnlyHint=True`.

```
=== list_documents {}                         isError=False  0.01s
[{"doc_id": "clean_application", "filename": "clean_application.txt", "characters": 345},
 {"doc_id": "risky_application", "filename": "risky_application.txt", "characters": 392}]

=== get_citation {"page": 190, "section": "Credit History"}   isError=False  0.01s
{"citation": "HUD Handbook 4000.1, p.190", "page": 190,
 "sections": ["Evaluating Credit History (TOTAL)"], "text": "three months of payments. ..."}

=== get_citation {"page": 5}                   isError=True
Error executing tool get_citation: page 5 is not in the corpus (available pages: 182-273)

=== search_guidelines {"query": "", "top_k": 5}  isError=True
Error executing tool search_guidelines: query must not be empty

=== analyze_document {"doc_id": "../../.env"}   isError=True
Error executing tool analyze_document: doc_id must be lowercase letters, digits, '-' or '_' (see list_documents)
```
In-process (not via MCP) `search_guidelines("chapter 7 bankruptcy waiting period", 2)` returned pages `[264, 265]` in 14.2 s including the one-time index build.

**Not yet captured through the MCP client:** live `search_guidelines` and `analyze_document` calls (`python -m mcp_server.demo_client --live`). The OpenRouter account ran out of credits during this session (HTTP 402), so these were not verified end to end over MCP. Their logic is covered by unit tests with a fake retriever and analyzer (`tests/test_mcp_tools.py`).

## Known limitations
- Handbook text contains a few `�` characters where the PDF's curly apostrophes did not decode.
- `analyze_document` only analyzes `.txt` files in the documents folder, not arbitrary uploads (deliberate, to keep the tool read-only and path-safe).
