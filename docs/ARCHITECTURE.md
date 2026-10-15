# Architecture (audited from the repo)

## Module map
| Layer | Location | Notes |
|---|---|---|
| API | `backend/app/routers/` (`auth`, `analysis`, `streaming` SSE, `admin`, `health`, `integrations`) | prefix `/api/v1`; JWT / API-key auth in `dependencies.py` |
| Middleware | `backend/app/middleware/` | request logging, slowapi rate limit (Redis) |
| Agent graph | `backend/app/agent/orchestrator.py` | LangGraph nodes: `extract_facts` → `retrieve_context` (multi-query, page metadata) → `llm_analyze` (cites excerpt pages) → `validate_citations` (grounding check) → `calculate_risk` → `generate_output` |
| Agent tools | `backend/app/agent/tools.py` | regex fact extraction, query building, source retrieval, citation validation, severity-based scoring (no hard-coded underwriting thresholds) |
| Retrieval | `backend/app/services/vector_service.py` | corpus loader, chunker, Chroma, retriever |
| LLM / RAG | `backend/app/services/ai_service.py` | provider switch (OpenRouter/OpenAI/Azure/Anthropic/Ollama), LCEL RAG chain, JSON parsing |
| Documents | `backend/app/services/document_service.py` | PDF/DOCX/TXT text extraction |
| DB | `backend/app/models/database.py`, `alembic/` | async SQLAlchemy; SQLite (dev) / PostgreSQL |
| Eval | `backend/evaluation/retrieval.py`, `backend/tests/golden/questions.jsonl` | golden-set retrieval metrics |
| Corpus | `backend/data/hud_4000_1_borrower_underwriting.jsonl` | HUD Handbook 4000.1 II.A.4 (92 pages, public domain), built by `scripts/build_hud_corpus.py` |

## External dependencies
OpenRouter (chat + embeddings, `LLM_PROVIDER=openrouter`), ChromaDB (persistent), PostgreSQL/SQLite, Redis (rate limit), optional Salesforce / Snowflake / Azure (`integrations/`).

## Commands
```bash
cd backend
pytest                      # unit + API tests (no network), coverage: pytest --cov=app --cov=evaluation
pytest -m eval -s           # retrieval eval on the golden set (needs OPENROUTER_API_KEY)
ruff check .                # lint
docker-compose up --build   # full stack
```

## Hard-to-test spots and what was done
- Module-level `_retriever` / `_rag_chain` globals: `build_vector_db` now takes explicit params (chunk size/overlap, k, embeddings, persist dir, collection) so tests and experiments inject their own.
- Chunk size, overlap and embedding model were hard-coded: now in `Settings` (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`).
- Startup re-embedded and duplicated chunks into the persistent collection on every restart: now reused unless `force_rebuild`.
- `admin.py` called `build_vector_db(force_rebuild=True)`, which the old signature did not accept: fixed by the new signature.

## Citation grounding (how answers are cited)
1. `retrieve_context` builds several focused queries (base topics + facts + topics seen in the document) and retrieves excerpts with their handbook page and section.
2. `llm_analyze` sees each excerpt under a label like `[HUD Handbook 4000.1, p.192]` and must copy one label into each flag's `guideline_reference`.
3. `validate_citations` keeps a citation only if the named page was actually retrieved; otherwise the flag is marked `Uncited` and its confidence capped at 0.4. Ungrounded flags do not raise the score.
4. If the LLM step fails or returns non-JSON, the recommendation fails safe to `MANUAL_REVIEW`, never `APPROVE`.

## Known issues (not yet fixed)
- Chunks never span pages (page is the citation unit); rules continued across a page break can be split.
- `datetime.utcnow()` deprecation warnings (Python 3.12+).
- Section labels are approximate (nearest heading detected in the PDF text).
