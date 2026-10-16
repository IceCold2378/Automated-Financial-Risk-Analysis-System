# Upgrade notes

Four stacked branches, merge in this order. All numbers were measured in real runs; see the linked docs.

## 1. `feat/testing-ci-eval`: corpus, tests, CI
- Replaced the 8-rule guidelines file (one chunk, so retrieval was unmeasurable) with HUD Handbook 4000.1 II.A.4 (92 pages, public domain) carrying page/section metadata (`backend/scripts/build_hud_corpus.py`).
- `build_vector_db` is parametrized and injectable (chunk size/overlap, k, embeddings, persist dir), reuses a persisted index instead of re-embedding and duplicating chunks, and supports `force_rebuild` (fixes the admin route calling it with an unsupported argument).
- 27-question golden set + `evaluation/retrieval.py` (hit rate, MRR, latency); offline test that every expected page contains its evidence phrase.
- Tests 74 -> 99, coverage 53% -> 58%; ruff config; CI runs lint + coverage floor, retrieval eval on manual dispatch.
- Docs: `docs/ARCHITECTURE.md`, `docs/BASELINE.md`.

## 2. `feat/mlflow-tracking`: experiment tracking
- `evaluation/run_eval.py` logs params, hit rate, MRR, latency and a failed-cases CSV to MLflow; `--sweep` runs a preset grid.
- Result and caveats: `docs/EXPERIMENTS.md`.

## 3. `feat/hud-citations`: cited agent
- Agent retrieves with several focused queries, the LLM cites the handbook page of each supporting excerpt, and a new `validate_citations` node rejects citations to pages that were not retrieved. Hard-coded thresholds (credit 680, DTI 40%) removed; scoring uses grounded-flag severity; LLM failure fails safe to `MANUAL_REVIEW`.
- Defaults changed to `baai/bge-large-en-v1.5`, chunk 500/100 (retrieval hit rate 92.6% -> 100%, MRR 0.870 -> 0.910 on the golden set). The Chroma collection name now encodes model and chunking, so changing them rebuilds the index.
- `evaluation/e2e.py` + 11 synthetic applications: end-to-end citation hit rate 70-80%, grounded-citation rate 93.5-100%.

## 4. `feat/mcp-server`: MCP server
- `mcp_server/`: four read-only tools (`search_guidelines`, `get_citation`, `list_documents`, `analyze_document`) over stdio, input validation, allowlisted document ids. 20 unit tests. See `docs/MCP.md`.
- Not yet verified over MCP with live API calls (the OpenRouter account ran out of credits); stated in the doc.

## Known limitations
- Small evaluation sets (27 questions, 10 applications); the configuration was chosen on the same data it is scored on.
- Upgrading requires `alembic upgrade head` on a fresh SQLite database; the old local `chroma_db` is not reused (new collection name).
- The CI workflow has not run on GitHub yet.
