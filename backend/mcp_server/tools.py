"""Read-only tool implementations behind the MCP server.

Everything here reuses the existing service layer (vector_service, agent orchestrator);
nothing writes to the database, the vector store, or the filesystem.
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

BACKEND_DIR = Path(__file__).resolve().parents[1]
CITATION_LABEL = "HUD Handbook 4000.1"
MAX_QUERY_CHARS = 500
MAX_TOP_K = 10
_DOC_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def docs_dir() -> Path:
    return Path(os.getenv("MCP_DOCS_DIR", BACKEND_DIR / "data" / "test_samples"))


@lru_cache(maxsize=1)
def _corpus() -> dict[int, list[dict]]:
    """Handbook pages from the configured corpus: page -> [{section, text}]."""
    pages: dict[int, list[dict]] = {}
    with open(get_settings().effective_guidelines_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pages.setdefault(rec["page"], []).append({"section": rec.get("section", ""), "text": rec["text"]})
    return pages


def search_guidelines(query: str, top_k: int = 5, retriever=None) -> list[dict]:
    """Semantic search over the guideline corpus. Returns excerpts with citation labels."""
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query too long ({len(query)} > {MAX_QUERY_CHARS} characters)")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    if retriever is None:
        from app.services.vector_service import get_retriever
        retriever = get_retriever()
    docs = retriever.vectorstore.similarity_search(query, k=top_k)
    return [{
        "citation": f"{CITATION_LABEL}, p.{d.metadata.get('page')}",
        "page": d.metadata.get("page"),
        "section": d.metadata.get("section", ""),
        "text": d.page_content,
    } for d in docs]


def get_citation(page: int, section: str | None = None) -> dict:
    """Return the full text of one handbook page, optionally filtered to a section name."""
    corpus = _corpus()
    if page not in corpus:
        lo, hi = min(corpus), max(corpus)
        raise ValueError(f"page {page} is not in the corpus (available pages: {lo}-{hi})")
    records = corpus[page]
    if section:
        records = [r for r in records if section.lower() in r["section"].lower()]
        if not records:
            raise ValueError(f"no section matching '{section}' on page {page}; "
                             f"sections there: {sorted({r['section'] for r in corpus[page]})}")
    return {
        "citation": f"{CITATION_LABEL}, p.{page}",
        "page": page,
        "sections": sorted({r["section"] for r in records}),
        "text": "\n".join(r["text"] for r in records),
    }


def list_documents() -> list[dict]:
    """Applications available for analysis (.txt files in the documents directory)."""
    d = docs_dir()
    if not d.is_dir():
        return []
    return [{"doc_id": p.stem, "filename": p.name, "characters": len(p.read_text(encoding="utf-8", errors="ignore"))}
            for p in sorted(d.glob("*.txt")) if _DOC_ID.match(p.stem)]


def _resolve_document(doc_id: str) -> Path:
    if not _DOC_ID.match(doc_id or ""):
        raise ValueError("doc_id must be lowercase letters, digits, '-' or '_' (see list_documents)")
    known = {d["doc_id"] for d in list_documents()}
    if doc_id not in known:
        raise ValueError(f"unknown doc_id '{doc_id}'; available: {sorted(known)}")
    return docs_dir() / f"{doc_id}.txt"


async def analyze_document(doc_id: str, analyzer=None) -> dict:
    """Run the cited-analysis agent on a listed document. Does not persist anything."""
    path = _resolve_document(doc_id)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if analyzer is None:
        from app.agent.orchestrator import run_agent_analysis as analyzer
    result, trace = await analyzer(document_text=text, filename=path.name)
    flags = result.risk_flags
    return {
        "doc_id": doc_id,
        "summary": result.summary,
        "risk_score": result.overall_risk_score,
        "risk_level": str(getattr(result.overall_risk_level, "value", result.overall_risk_level)),
        "recommendation": result.recommendation,
        "flags": [{"category": f.category, "severity": str(getattr(f.severity, "value", f.severity)),
                   "description": f.description, "citation": f.guideline_reference,
                   "confidence": f.confidence} for f in flags],
        "grounded_flags": sum(1 for f in flags if not f.guideline_reference.startswith("Uncited")),
        "processing_time_seconds": result.processing_time_seconds,
    }
