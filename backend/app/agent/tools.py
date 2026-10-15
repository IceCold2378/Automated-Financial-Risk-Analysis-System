"""
Agent tools — callable functions the LangGraph agent can invoke.

Each tool performs a focused task (guideline retrieval, risk calculation,
compliance checking) and returns structured data back to the agent state.
"""

import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


# ── Tool: Retrieve Guidelines ─────────────────────────────────────

def retrieve_guidelines(query: str, retriever: Any) -> list[str]:
    """
    Search the vector DB for guideline chunks relevant to the query.

    Args:
        query: The search query (usually the document text or a focused question)
        retriever: LangChain retriever instance

    Returns:
        List of retrieved guideline text chunks
    """
    logger.info("Tool:retrieve_guidelines — query length=%d chars", len(query))
    try:
        # Truncate query to avoid embedding limits
        truncated = query[:4000] if len(query) > 4000 else query
        docs = retriever.invoke(truncated)
        chunks = [doc.page_content for doc in docs]
        logger.info("Retrieved %d guideline chunks", len(chunks))
        return chunks
    except Exception as e:
        logger.error("Guideline retrieval failed: %s", e)
        return []


# ── Tool: Extract Document Facts ──────────────────────────────────

def extract_document_facts(document_text: str) -> dict:
    """
    Extract structured facts from the loan application text using regex
    and pattern matching. This is a deterministic pre-processing step
    before the LLM analysis.

    Returns a dict of extracted fields (may have None values if not found).
    """
    facts: dict[str, Any] = {
        "credit_score": None,
        "dti_ratio": None,
        "ltv_ratio": None,
        "loan_amount": None,
        "down_payment": None,
        "annual_income": None,
        "applicant_age": None,
        "employment_years": None,
        "bankruptcy_mentioned": False,
        "property_value": None,
        "loan_type": None,
        "first_time_buyer": None,
    }

    text_lower = document_text.lower()

    # Credit score
    match = re.search(r'credit\s*score[:\s]+(?:of\s+)?(\d{3})', text_lower)
    if match:
        facts["credit_score"] = int(match.group(1))

    # DTI ratio
    match = re.search(r'(?:dti|debt.to.income)\s*(?:ratio)?[:\s]+(?:is\s+)?(\d+\.?\d*)%?', text_lower)
    if match:
        facts["dti_ratio"] = float(match.group(1))

    # LTV ratio
    match = re.search(r'(?:ltv|loan.to.value)[:\s]*(\d+\.?\d*)%?', text_lower)
    if match:
        facts["ltv_ratio"] = float(match.group(1))

    # Loan amount
    match = re.search(r'loan\s*amount[:\s]*\$?([\d,]+\.?\d*)', text_lower)
    if match:
        facts["loan_amount"] = float(match.group(1).replace(',', ''))

    # Annual income
    match = re.search(r'(?:annual|yearly)\s*income[:\s]*\$?([\d,]+\.?\d*)', text_lower)
    if match:
        facts["annual_income"] = float(match.group(1).replace(',', ''))

    # Down payment
    match = re.search(r'down\s*payment[:\s]*\$?([\d,]+\.?\d*)', text_lower)
    if match:
        facts["down_payment"] = float(match.group(1).replace(',', ''))

    # Age
    match = re.search(r'age[:\s]*(\d{2})', text_lower)
    if match:
        facts["applicant_age"] = int(match.group(1))

    # Bankruptcy
    if 'bankruptcy' in text_lower:
        facts["bankruptcy_mentioned"] = True

    # Property value
    match = re.search(r'property\s*value[:\s]*\$?([\d,]+\.?\d*)', text_lower)
    if match:
        facts["property_value"] = float(match.group(1).replace(',', ''))

    # First-time buyer
    if 'first.time' in text_lower or 'first time' in text_lower:
        facts["first_time_buyer"] = True

    logger.info("Extracted facts: %s",
                {k: v for k, v in facts.items() if v is not None and v is not False})
    return facts


# ── Tool: Retrieve Cited Sources ──────────────────────────────────

CITATION_LABEL = "HUD Handbook 4000.1"

# Query expansion only (what to look up); no thresholds or pass/fail rules live here.
BASE_QUERIES = [
    "credit history requirements, late payments, bankruptcy, foreclosure",
    "debt-to-income ratio and evaluating liabilities and debts",
    "effective income, employment history and stability of income",
    "assets, reserves, gift funds and minimum required investment",
]
TOPIC_QUERIES = {
    "bankruptcy": "bankruptcy waiting period after discharge",
    "foreclosure": "foreclosure waiting period",
    "collection": "collection accounts and disputed derogatory credit",
    "self-employ": "self-employment income requirements",
    "commission": "commission income calculation",
    "overtime": "overtime and bonus income",
    "student loan": "student loan monthly payment calculation",
    "child support": "child support and alimony income",
    "gift": "gift funds documentation",
    "rental": "rental income calculation",
    "social security": "social security and disability income",
    "retire": "retirement account and pension income",
}


def build_retrieval_queries(facts: dict, document_text: str) -> list[str]:
    """Build focused retrieval queries from extracted facts and topics seen in the document."""
    queries = list(BASE_QUERIES)
    if facts.get("credit_score"):
        queries.append("minimum credit score requirements")
    if facts.get("dti_ratio"):
        queries.append("total debt-to-income qualifying ratio")
    if facts.get("loan_amount") or facts.get("down_payment"):
        queries.append("minimum required investment and down payment")
    lowered = document_text.lower()
    queries += [q for kw, q in TOPIC_QUERIES.items() if kw in lowered]
    return queries


def retrieve_sources(queries: list[str], retriever: Any, max_chunks: int = 12) -> list[dict]:
    """Run all queries (concurrently), dedupe, and return chunks with page/section metadata."""
    try:
        results = retriever.batch([q[:2000] for q in queries])
    except Exception as e:
        logger.error("Guideline retrieval failed: %s", e)
        return []
    seen, sources = set(), []
    # Interleave so every query contributes its best hit before any query's lower hits.
    for rank in range(max((len(r) for r in results), default=0)):
        for docs in results:
            if rank < len(docs):
                d = docs[rank]
                key = (d.metadata.get("page"), d.page_content[:80])
                if key not in seen:
                    seen.add(key)
                    sources.append({"page": d.metadata.get("page"), "section": d.metadata.get("section", ""),
                                    "text": d.page_content})
    return sources[:max_chunks]


def format_sources(sources: list[dict]) -> str:
    """Render sources for the prompt, each headed by the citation label the LLM must copy."""
    return "\n\n".join(f"[{CITATION_LABEL}, p.{s['page']}] ({s['section']})\n{s['text']}" for s in sources)


# ── Tool: Validate Citations ──────────────────────────────────────

_PAGE_RE = re.compile(r"p(?:p|age)?\.?\s*(\d{1,4})", re.I)


def validate_citations(flags: list[dict], sources: list[dict]) -> tuple[list[dict], dict]:
    """
    Ground every risk flag's citation in the retrieved sources.

    A citation is grounded when a page it names was actually retrieved. Grounded citations are
    normalized to "HUD Handbook 4000.1, p.N (section)"; ungrounded ones are marked and their
    confidence capped, so unsupported claims never look authoritative.
    """
    by_page = {s["page"]: s["section"] for s in sources}
    out, grounded = [], 0
    for f in flags:
        f = dict(f)
        pages = [int(p) for p in _PAGE_RE.findall(f.get("guideline_reference", "") or "")]
        hit = next((p for p in pages if p in by_page), None)
        if hit is not None:
            f["guideline_reference"] = f"{CITATION_LABEL}, p.{hit} ({by_page[hit]})"
            f["grounded"] = True
            grounded += 1
        else:
            f["guideline_reference"] = "Uncited: no supporting handbook excerpt was retrieved"
            f["confidence"] = min(float(f.get("confidence", 0.5)), 0.4)
            f["grounded"] = False
        out.append(f)
    stats = {"flags": len(out), "grounded": grounded,
             "grounded_rate": (grounded / len(out)) if out else 1.0}
    return out, stats


# ── Tool: Risk Score & Recommendation ─────────────────────────────

SEVERITY_WEIGHT = {"low": 5, "moderate": 10, "high": 20, "critical": 35}


def risk_level_for(score: int) -> str:
    return "low" if score <= 25 else "moderate" if score <= 50 else "high" if score <= 75 else "critical"


def calculate_risk_score(flags: list[dict], llm_score: int | None = None) -> tuple[int, str]:
    """
    Score = severity-weighted sum of *grounded* flags, averaged with the LLM's own score when
    available. Ungrounded flags do not raise the score.
    """
    weighted = min(100, sum(SEVERITY_WEIGHT.get(f.get("severity", "moderate"), 10)
                            for f in flags if f.get("grounded")))
    score = weighted if llm_score is None else round((weighted + min(100, max(0, llm_score))) / 2)
    return score, risk_level_for(score)


def determine_recommendation(risk_score: int, flags: list[dict]) -> str:
    """APPROVE / DENY / MANUAL_REVIEW from the score and grounded flag severities."""
    grounded = [f for f in flags if f.get("grounded")]
    if any(f.get("severity") == "critical" for f in grounded) or risk_score >= 76:
        return "DENY"
    if any(f.get("severity") in ("high", "critical") for f in flags) or risk_score >= 26:
        return "MANUAL_REVIEW"
    return "APPROVE"
