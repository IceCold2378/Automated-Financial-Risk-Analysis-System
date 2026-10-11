# 🛡️ Automated Financial Risk Analysis System

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

## 🧪 Testing

```bash
cd backend
pip install pytest pytest-asyncio httpx aiosqlite
pytest tests/ -v
```
