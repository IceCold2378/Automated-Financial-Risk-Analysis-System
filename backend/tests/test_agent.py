"""
Tests for the agentic analysis pipeline.
"""


from app.agent.state import AgentConfig, AgentTrace, StepStatus
from app.agent.orchestrator import node_calculate_risk, node_llm_analyze
from app.agent.tools import (
    build_retrieval_queries,
    calculate_risk_score,
    determine_recommendation,
    extract_document_facts,
    format_sources,
    retrieve_sources,
    validate_citations,
)


# ══════════════════════════════════════════════════════════════════
#  Tool Tests — Deterministic, no mocking needed
# ══════════════════════════════════════════════════════════════════

class TestExtractDocumentFacts:
    """Test the regex-based fact extractor."""

    def test_extracts_credit_score(self):
        text = "The applicant has a credit score of 720."
        facts = extract_document_facts(text)
        assert facts["credit_score"] == 720

    def test_extracts_dti_ratio(self):
        text = "The debt-to-income ratio is 35%."
        facts = extract_document_facts(text)
        assert facts["dti_ratio"] == 35.0

    def test_extracts_loan_amount(self):
        text = "Loan amount: $350,000"
        facts = extract_document_facts(text)
        assert facts["loan_amount"] == 350000.0

    def test_detects_bankruptcy(self):
        text = "The applicant declared bankruptcy in 2019."
        facts = extract_document_facts(text)
        assert facts["bankruptcy_mentioned"] is True

    def test_no_bankruptcy_when_absent(self):
        text = "The applicant has a clean financial history."
        facts = extract_document_facts(text)
        assert facts["bankruptcy_mentioned"] is False

    def test_extracts_multiple_facts(self):
        text = """
        Applicant: John Doe, Age: 32
        Credit Score: 650
        Annual Income: $85,000
        Loan Amount: $400,000
        DTI: 42%
        Down Payment: $50,000
        """
        facts = extract_document_facts(text)
        assert facts["credit_score"] == 650
        assert facts["annual_income"] == 85000.0
        assert facts["loan_amount"] == 400000.0
        assert facts["dti_ratio"] == 42.0
        assert facts["down_payment"] == 50000.0
        assert facts["applicant_age"] == 32

    def test_handles_empty_document(self):
        facts = extract_document_facts("")
        assert facts["credit_score"] is None
        assert facts["bankruptcy_mentioned"] is False


SOURCES = [
    {"page": 188, "section": "Evaluating Credit History (TOTAL)", "text": "Disputed derogatory credit..."},
    {"page": 192, "section": "Evaluating Liabilities and Debts (TOTAL)", "text": "Late mortgage payments..."},
]


class _Doc:
    def __init__(self, page, text):
        self.page_content, self.metadata = text, {"page": page, "section": "S"}


class _BatchRetriever:
    def batch(self, queries):
        return [[_Doc(188, "a" * 90), _Doc(192, "b" * 90)] for _ in queries]


class TestRetrieveSources:
    def test_queries_follow_facts_and_topics(self):
        qs = build_retrieval_queries({"credit_score": 700, "dti_ratio": 45.0},
                                     "Applicant is self-employed and had a bankruptcy.")
        assert "minimum credit score requirements" in qs
        assert "total debt-to-income qualifying ratio" in qs
        assert any("self-employment" in q for q in qs) and any("bankruptcy" in q for q in qs)

    def test_dedupes_and_keeps_page_metadata(self):
        sources = retrieve_sources(["q1", "q2", "q3"], _BatchRetriever())
        assert [s["page"] for s in sources] == [188, 192]

    def test_respects_max_chunks(self):
        assert len(retrieve_sources(["q"], _BatchRetriever(), max_chunks=1)) == 1

    def test_retrieval_failure_returns_empty(self):
        class Boom:
            def batch(self, q):
                raise RuntimeError("embedding api down")
        assert retrieve_sources(["q"], Boom()) == []

    def test_format_sources_has_citation_labels(self):
        out = format_sources(SOURCES)
        assert "[HUD Handbook 4000.1, p.188]" in out and "[HUD Handbook 4000.1, p.192]" in out


class TestValidateCitations:
    def test_grounded_citation_is_normalized(self):
        flags, stats = validate_citations(
            [{"severity": "high", "guideline_reference": "HUD Handbook 4000.1, p.192", "confidence": 0.9}], SOURCES)
        assert flags[0]["grounded"] and "p.192" in flags[0]["guideline_reference"]
        assert "Evaluating Liabilities" in flags[0]["guideline_reference"]
        assert stats == {"flags": 1, "grounded": 1, "grounded_rate": 1.0}

    def test_hallucinated_page_is_rejected_and_confidence_capped(self):
        flags, stats = validate_citations(
            [{"severity": "high", "guideline_reference": "HUD Handbook 4000.1, p.999", "confidence": 0.95}], SOURCES)
        assert not flags[0]["grounded"] and flags[0]["confidence"] == 0.4
        assert flags[0]["guideline_reference"].startswith("Uncited")
        assert stats["grounded_rate"] == 0.0

    def test_empty_reference_is_ungrounded(self):
        flags, _ = validate_citations([{"severity": "low", "guideline_reference": ""}], SOURCES)
        assert not flags[0]["grounded"]

    def test_no_flags_is_fully_grounded(self):
        assert validate_citations([], SOURCES)[1]["grounded_rate"] == 1.0


class TestRiskAndRecommendation:
    def test_ungrounded_flags_do_not_raise_score(self):
        score, level = calculate_risk_score([{"severity": "critical", "grounded": False}])
        assert score == 0 and level == "low"

    def test_grounded_flags_sum_by_severity(self):
        flags = [{"severity": "high", "grounded": True}, {"severity": "moderate", "grounded": True}]
        assert calculate_risk_score(flags) == (30, "moderate")

    def test_score_averages_with_llm_score(self):
        assert calculate_risk_score([{"severity": "high", "grounded": True}], llm_score=60)[0] == 40

    def test_score_is_capped_and_levels(self):
        flags = [{"severity": "critical", "grounded": True}] * 5
        assert calculate_risk_score(flags) == (100, "critical")
        assert calculate_risk_score([], llm_score=-5)[0] == 0

    def test_approve_when_clean(self):
        assert determine_recommendation(10, []) == "APPROVE"

    def test_deny_on_grounded_critical(self):
        assert determine_recommendation(40, [{"severity": "critical", "grounded": True}]) == "DENY"

    def test_deny_on_very_high_score(self):
        assert determine_recommendation(80, []) == "DENY"

    def test_manual_review_on_ungrounded_high_flag(self):
        assert determine_recommendation(5, [{"severity": "high", "grounded": False}]) == "MANUAL_REVIEW"

    def test_manual_review_on_moderate_score(self):
        assert determine_recommendation(30, []) == "MANUAL_REVIEW"


class TestAgentNodes:
    def test_llm_failure_fails_safe_to_manual_review(self):
        out = node_calculate_risk({"llm_ok": False, "risk_flags": [], "trace": AgentTrace().to_dict()})
        assert out["recommendation"] == "MANUAL_REVIEW" and out["risk_score"] == 50

    def test_llm_node_fails_without_sources(self):
        out = node_llm_analyze({"document_text": "x", "retrieved_sources": [], "trace": AgentTrace().to_dict()})
        assert out["llm_ok"] is False and "grounding" in out["error"]

    def test_llm_node_parses_flags(self, monkeypatch):
        import json
        from app.services import ai_service

        class Chain:
            def invoke(self, payload):
                assert "p.188" in payload["context"]
                return json.dumps({"summary": "s", "overall_risk_score": 60, "risk_flags": [
                    {"category": "Credit", "severity": "high", "guideline_reference": "p.188"}]})

        monkeypatch.setattr(ai_service, "get_analysis_chain", lambda: Chain())
        out = node_llm_analyze({"document_text": "app", "retrieved_sources": SOURCES, "trace": AgentTrace().to_dict()})
        assert out["llm_ok"] and out["llm_risk_score"] == 60 and len(out["risk_flags"]) == 1

    def test_llm_node_non_json_is_not_ok(self, monkeypatch):
        from app.services import ai_service

        class Chain:
            def invoke(self, payload):
                return "sorry, plain text"

        monkeypatch.setattr(ai_service, "get_analysis_chain", lambda: Chain())
        out = node_llm_analyze({"document_text": "app", "retrieved_sources": SOURCES, "trace": AgentTrace().to_dict()})
        assert out["llm_ok"] is False and out["risk_flags"] == []


class TestAgentTrace:
    """Test audit trail recording."""

    def test_trace_add_step(self):
        trace = AgentTrace()
        step = trace.add_step("extract_facts")
        assert step.step_number == 1
        assert step.node_name == "extract_facts"
        assert step.status == StepStatus.PENDING

    def test_trace_multiple_steps(self):
        trace = AgentTrace()
        trace.add_step("step_1")
        trace.add_step("step_2")
        trace.add_step("step_3")
        assert len(trace.steps) == 3
        assert trace.steps[2].step_number == 3

    def test_trace_serialization(self):
        trace = AgentTrace()
        step = trace.add_step("test_node")
        step.status = StepStatus.COMPLETE
        step.duration_ms = 42.5

        d = trace.to_dict()
        assert len(d["steps"]) == 1
        assert d["steps"][0]["status"] == "complete"
        assert d["steps"][0]["duration_ms"] == 42.5


class TestAgentConfig:
    """Test agent runtime configuration."""

    def test_defaults(self):
        config = AgentConfig()
        assert config.max_iterations == 5
        assert config.enable_tools is True
        assert config.temperature == 0.1

    def test_custom_config(self):
        config = AgentConfig(max_iterations=10, temperature=0.5)
        assert config.max_iterations == 10
        assert config.temperature == 0.5
