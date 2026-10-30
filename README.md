# 🛡️ Intelligent Underwriting Assistant — v2.0 Pro

A commercial-grade, full-stack AI platform that automates loan application risk analysis using Large Language Models. Built for underwriting professionals who need fast, consistent, and auditable risk assessments.

---

## 🎯 What It Does

Upload a loan application (PDF, DOCX, or TXT) and receive a structured AI-powered risk analysis including:

- **Risk Score** (0-100) with color-coded severity levels
- **Individual Risk Flags** with confidence scores and guideline references
- **Recommendation** (Approve / Deny / Manual Review)
- **Full Audit Trail** — every analysis is stored with timestamp, user, and metadata

*(Demo GIF coming soon).*

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19 + Vite, Vanilla CSS (dark-mode glassmorphism) |
| **Backend** | FastAPI (Python), async architecture |
| **AI/LLM** | LangGraph + LangChain + Ollama/Mistral (OpenAI GPT-4 optional) |
| **Vector DB** | ChromaDB (persistent) for RAG |
| **Database** | Alembic + PostgreSQL (production) / SQLite (development) |
| **Auth** | API Keys & JWT (bcrypt + PyJWT) |
| **Containerization** | Docker & Docker Compose |

---

## 📁 Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py             # Environment config (pydantic-settings)
│   │   ├── dependencies.py       # Auth & RBAC injection
│   │   ├── models/               # SQLAlchemy async models & Schemas
│   │   ├── routers/              # API Endpoints
│   │   ├── services/             # Background Task Brokers & External services
│   │   ├── integrations/         # Salesforce, Azure, Snowflake logic
│   │   └── agent/                # LangGraph Stateful AI Orchestrator
│   ├── alembic/                  # Database migration management
│   ├── tests/                    # pytest test suite
│   ├── data/guidelines.txt       # Underwriting guidelines
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Router + protected routes
│   │   ├── api.js                # Remote EventSource (SSE) hooks
│   │   ├── index.css             # Design system
│   │   └── pages/                # Admin Panel, Tracking, Analysis
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 🚀 Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Ollama](https://ollama.com/) installed with the `mistral` model (`ollama pull mistral`)
- [Node.js 20+](https://nodejs.org/) (for frontend development)

### Quick Start (Docker)

```bash
# 1. Clone repository
git clone https://github.com/IceCold2378/Intelligent-Underwriting-Assistant.git
cd Intelligent-Underwriting-Assistant

# 2. Copy environment config
cp .env.example backend/.env

# 3. Launch all services
docker-compose up --build

# 4. Access the app
#    Frontend: http://localhost:5173
#    API Docs: http://localhost:8000/api/v1/docs
#    Health:   http://localhost:8000/api/v1/health
```

### Local Development (Without Docker)

```bash
# Backend
cd backend
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

---

## 🔑 API Endpoints

All endpoints are prefixed with `/api/v1`.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | ❌ | System health check |
| `POST` | `/auth/register` | ❌ | Create account |
| `POST` | `/auth/login` | ❌ | Get JWT token |
| `GET` | `/stream/task/{id}`| ✅ | **SSE** Live trace agent execution stream |
| `POST` | `/analysis/task` | ✅ | Upload & analyze document |
| `GET` | `/analysis/history`| ✅ | Paginated analysis history |
| `GET` | `/admin/metrics` | 👑 | System metrics & vector logs |

---

## ⚙️ Configuration

All settings are managed via environment variables (`.env` file). See `.env.example` for all options.

---

## 🧪 Testing & Evaluation

```bash
cd backend
pip install -r requirements.txt
pytest                                   # fast unit/API tests, no network
pytest --cov=app --cov=evaluation        # with coverage
pytest -m eval -s                        # retrieval hit-rate on the golden set (needs OPENROUTER_API_KEY)
ruff check .                             # lint
```

CI (`.github/workflows/ci.yml`) runs ruff and pytest with a coverage floor on every push/PR; the eval job runs on manual dispatch.
The guideline corpus is HUD Handbook 4000.1, Section II.A.4 (public domain), rebuilt with `python scripts/build_hud_corpus.py <pdf>`.
The golden set lives in `backend/tests/golden/questions.jsonl`. Details and measured numbers: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/BASELINE.md](docs/BASELINE.md).

### Experiment tracking (MLflow)

```bash
cd backend
python -m evaluation.run_eval --sweep                 # run the preset grid, logs to mlflow.db
python -m evaluation.run_eval --chunk-size 500 --chunk-overlap 100 --k 5   # single config
mlflow ui --backend-store-uri sqlite:///mlflow.db     # compare runs
```
Results and the chosen configuration: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

### Cited analysis

The agent retrieves HUD Handbook excerpts, the LLM must cite the page of the excerpt behind each flag, and a validation step rejects citations to pages that were not retrieved. Measure end-to-end citation quality with `python -m evaluation.e2e --mlflow` (see [docs/BASELINE.md](docs/BASELINE.md)).

### MCP server

The retrieval and analysis pipeline is also available as a [Model Context Protocol](https://modelcontextprotocol.io) server, so any MCP-compatible client (a desktop AI assistant, an IDE agent, or your own script) can use it without knowing anything about this codebase. It is separate from the web app and runs locally over stdio.

| Tool | What it does |
|---|---|
| `search_guidelines(query, top_k)` | Semantic search over the HUD Handbook; returns passages with page citations |
| `get_citation(page, section?)` | Full text of one handbook page, to verify a citation against the source |
| `list_documents()` | Loan applications available for analysis |
| `analyze_document(doc_id)` | Full cited risk analysis: score, recommendation and flags, each tied to a handbook page |

All tools are read-only; inputs are validated and `analyze_document` only accepts documents returned by `list_documents`.

**Run and connect**

```bash
cd backend
python -m mcp_server.server            # stdio server (needs OPENROUTER_API_KEY in backend/.env)
python -m mcp_server.demo_client --live   # quick end-to-end check using a real MCP client
```

To use it from an MCP client, add the server to the client's configuration (the exact file depends on the client):

```json
{ "mcpServers": { "underwriting-assistant": {
    "command": "<repo>/venv/Scripts/python.exe",
    "args": ["-m", "mcp_server.server"],
    "cwd": "<repo>/backend" } } }
```

**Things you can then ask the connected assistant**
- "Search the guidelines: how long after a Chapter 7 bankruptcy discharge can a borrower qualify?" (returns pages 264-265 with the two-year rule)
- "Show me the full text of handbook page 190."
- "List the available applications, then analyze `risky_application` and explain each flag using its cited page."

Example output from a real run (single run; model output varies):

```
analyze_document("risky_application") -> score 92, critical, DENY, 6 of 6 flags cite a retrieved page
  critical | Credit History        | HUD Handbook 4000.1, p.190   (bankruptcy discharged 14 months ago)
  high     | Employment Stability | HUD Handbook 4000.1, p.209   (self-employed 14 months)
  high     | Liabilities and Debts| HUD Handbook 4000.1, p.193 / p.196   (deferred student loan)
```

Tool schemas, security notes, a Windows startup note and the full captured calls: [docs/MCP.md](docs/MCP.md).
