"""
Agent tools — callable functions the LangGraph agent can invoke.

Each tool performs a focused task (guideline retrieval, risk calculation,
compliance checking) and returns structured data back to the agent state.
"""

import logging
import re
from typing import Any

from app.models.schemas import RiskLevel

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


# ── Tool: Calculate Risk Score ────────────────────────────────────

def calculate_risk_score(facts: dict, compliance_results: list[dict]) -> tuple[int, str]:
    """
    Deterministic risk score calculation from extracted facts and compliance results.

    Returns:
        (score 0-100, risk_level string)
    """
    score = 0
    penalties = []

    # Credit score penalties
    credit = facts.get("credit_score")
    if credit is not None:
        if credit < 580:
            score += 35
            penalties.append(f"Very low credit score ({credit})")
        elif credit < 620:
            score += 25
            penalties.append(f"Poor credit score ({credit})")
        elif credit < 680:
            score += 15
            penalties.append(f"Below threshold credit score ({credit})")
        elif credit < 720:
            score += 5
    else:
        score += 10
        penalties.append("Credit score not provided")

    # DTI ratio penalties
    dti = facts.get("dti_ratio")
    if dti is not None:
        if dti > 50:
            score += 25
            penalties.append(f"Very high DTI ({dti}%)")
        elif dti > 40:
            score += 15
            penalties.append(f"DTI exceeds 40% threshold ({dti}%)")
        elif dti > 35:
            score += 5
    else:
        score += 5
        penalties.append("DTI ratio not provided")

    # LTV ratio penalties
    ltv = facts.get("ltv_ratio")
    if ltv is not None:
        if ltv > 95:
            score += 20
            penalties.append(f"Very high LTV ({ltv}%)")
        elif ltv > 80:
            score += 10
            penalties.append(f"LTV exceeds 80% threshold ({ltv}%)")
    elif facts.get("loan_amount") and facts.get("property_value"):
        calculated_ltv = (facts["loan_amount"] / facts["property_value"]) * 100
        if calculated_ltv > 80:
            score += 10
            penalties.append(f"Calculated LTV exceeds 80% ({calculated_ltv:.1f}%)")

    # Bankruptcy flag
    if facts.get("bankruptcy_mentioned"):
        score += 15
        penalties.append("Bankruptcy history detected")

    # First-time buyer with high loan
    loan_amount = facts.get("loan_amount")
    if facts.get("first_time_buyer") and loan_amount and loan_amount > 500_000:
        score += 10
        penalties.append(f"First-time buyer with loan >${loan_amount:,.0f}")

    # Age check
    age = facts.get("applicant_age")
    if age is not None and age < 18:
        score += 20
        penalties.append(f"Applicant under 18 ({age})")

    # Additional penalties from compliance failures
    for result in compliance_results:
        if result.get("violated"):
            severity = result.get("severity", "moderate")
            if severity == "critical":
                score += 15
            elif severity == "high":
                score += 10
            elif severity == "moderate":
                score += 5

    # Clamp to 0-100
    score = min(100, max(0, score))

    # Determine risk level
    if score <= 25:
        level = RiskLevel.LOW
    elif score <= 50:
        level = RiskLevel.MODERATE
    elif score <= 75:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.CRITICAL

    logger.info("Risk calculation: score=%d, level=%s, penalties=%d",
                score, level.value, len(penalties))
    return score, level.value


# ── Tool: Check Compliance ────────────────────────────────────────

COMPLIANCE_RULES = [
    {
        "id": "CREDIT_SCORE_MIN",
        "name": "Minimum Credit Score",
        "description": "Credit score must be at least 680",
        "check": lambda facts: (
            facts.get("credit_score") is not None and facts["credit_score"] < 680
        ),
        "severity": "high",
        "guideline": "Loan Eligibility: minimum acceptable credit score is 680",
    },
    {
        "id": "APPLICANT_AGE",
        "name": "Minimum Age Requirement",
        "description": "Applicant must be at least 18 years old",
        "check": lambda facts: (
            facts.get("applicant_age") is not None and facts["applicant_age"] < 18
        ),
        "severity": "critical",
        "guideline": "Loan Eligibility: applicant must be at least 18 years old",
    },
    {
        "id": "DTI_RATIO",
        "name": "Debt-to-Income Ratio",
        "description": "DTI ratio must not exceed 40%",
        "check": lambda facts: (
            facts.get("dti_ratio") is not None and facts["dti_ratio"] > 40
        ),
        "severity": "high",
        "guideline": "Financial Ratios: DTI ratio must not exceed 40%",
    },
    {
        "id": "LTV_RATIO",
        "name": "Loan-to-Value Ratio",
        "description": "LTV ratio must not exceed 80% (min 20% down payment)",
        "check": lambda facts: (
            facts.get("ltv_ratio") is not None and facts["ltv_ratio"] > 80
        ),
        "severity": "high",
        "guideline": "Financial Ratios: LTV ratio must not exceed 80%",
    },
    {
        "id": "BANKRUPTCY",
        "name": "Bankruptcy History",
        "description": "Applicants with bankruptcy must be flagged for manual review",
        "check": lambda facts: facts.get("bankruptcy_mentioned", False),
        "severity": "high",
        "guideline": "Risk Flags: previous bankruptcy requires manual review",
    },
    {
        "id": "FIRST_TIME_BUYER_LIMIT",
        "name": "First-Time Buyer Loan Limit",
        "description": "First-time home buyer loan should not exceed $500,000",
        "check": lambda facts: (
            facts.get("first_time_buyer")
            and facts.get("loan_amount") is not None
            and facts["loan_amount"] > 500_000
        ),
        "severity": "moderate",
        "guideline": "Risk Flags: loan amount should not exceed $500,000 for first-time buyer",
    },
]


def check_compliance(facts: dict) -> list[dict]:
    """
    Run the extracted facts against all compliance rules.

    Returns a list of compliance check results, each containing:
    - rule_id, rule_name, violated (bool), severity, guideline_reference
    """
    results = []
    for rule in COMPLIANCE_RULES:
        try:
            violated = rule["check"](facts)
        except Exception:
            violated = False

        results.append({
            "rule_id": rule["id"],
            "rule_name": rule["name"],
            "description": rule["description"],
            "violated": violated,
            "severity": rule["severity"] if violated else "none",
            "guideline_reference": rule["guideline"],
        })

    violated_count = sum(1 for r in results if r["violated"])
    logger.info("Compliance check: %d/%d rules violated", violated_count, len(results))
    return results


# ── Tool: Determine Recommendation ───────────────────────────────

def determine_recommendation(risk_score: int, compliance_results: list[dict]) -> str:
    """
    Determine the final recommendation based on risk score and compliance results.

    Returns: "APPROVE", "DENY", or "MANUAL_REVIEW"
    """
    # Auto-deny conditions
    critical_violations = [
        r for r in compliance_results
        if r["violated"] and r["severity"] == "critical"
    ]
    if critical_violations:
        return "DENY"

    if risk_score >= 76:
        return "DENY"

    # Manual review conditions
    high_violations = [
        r for r in compliance_results
        if r["violated"] and r["severity"] == "high"
    ]
    if high_violations or risk_score >= 51:
        return "MANUAL_REVIEW"

    if risk_score >= 26:
        return "MANUAL_REVIEW"

    return "APPROVE"
