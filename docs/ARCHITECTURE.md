# Architecture (audited from the repo)

## Module map
| Layer | Location | Notes |
|---|---|---|
| API | `backend/app/routers/` (`auth`, `analysis`, `streaming` SSE, `admin`, `health`, `integrations`) | prefix `/api/v1`; JWT / API-key auth in `dependencies.py` |
| Middleware | `backend/app/middleware/` | request logging, slowapi rate limit (Redis) |
| Agent graph | `backend/app/agent/orchestrator.py` | LangGraph nodes: `extract_facts` → `retrieve_context` → `llm_analyze` → `check_compliance` → `calculate_risk` → `generate_output` |
| Rule tools | `backend/app/agent/tools.py` | regex fact extraction, **hard-coded** thresholds (credit 680, DTI 40%, LTV 80%), risk scoring |
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

## Known issues (not yet fixed)
- Final citations (`guideline_reference`) come from the hard-coded rules in `agent/tools.py`, not from retrieved HUD chunks; retrieval only feeds the LLM node.
- `node_llm_analyze` raises `NameError` when the LLM returns non-JSON (`llm_risk_flags` unbound); the exception handler masks it.
- `datetime.utcnow()` deprecation warnings (Python 3.12+).
