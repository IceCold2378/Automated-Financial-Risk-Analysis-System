"""
Tests for the agentic analysis pipeline.
"""


from app.agent.state import AgentConfig, AgentTrace, StepStatus
from app.agent.tools import (
    extract_document_facts,
    calculate_risk_score,
    check_compliance,
    determine_recommendation,
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


class TestCalculateRiskScore:
    """Test the deterministic risk scoring algorithm."""

    def test_low_risk_applicant(self):
        facts = {"credit_score": 780, "dti_ratio": 25.0, "ltv_ratio": 70.0}
        score, level = calculate_risk_score(facts, [])
        assert score <= 25
        assert level == "low"

    def test_high_risk_low_credit(self):
        facts = {"credit_score": 550, "dti_ratio": 30.0}
        score, level = calculate_risk_score(facts, [])
        assert score >= 35
        assert level in ("moderate", "high")

    def test_high_risk_high_dti(self):
        facts = {"credit_score": 700, "dti_ratio": 55.0}
        score, level = calculate_risk_score(facts, [])
        assert score >= 25

    def test_bankruptcy_adds_penalty(self):
        no_bankruptcy = {"credit_score": 700, "bankruptcy_mentioned": False}
        with_bankruptcy = {"credit_score": 700, "bankruptcy_mentioned": True}
        score_no, _ = calculate_risk_score(no_bankruptcy, [])
        score_with, _ = calculate_risk_score(with_bankruptcy, [])
        assert score_with > score_no

    def test_missing_data_adds_penalty(self):
        no_data = {}
        score, _ = calculate_risk_score(no_data, [])
        assert score >= 10  # Penalty for missing credit score + DTI

    def test_compliance_violations_increase_score(self):
        facts = {"credit_score": 700}
        violations = [
            {"violated": True, "severity": "critical"},
            {"violated": True, "severity": "high"},
        ]
        score, _ = calculate_risk_score(facts, violations)
        assert score >= 25


class TestCheckCompliance:
    """Test the compliance rule engine."""

    def test_credit_below_threshold(self):
        facts = {"credit_score": 600}
        results = check_compliance(facts)
        credit_rule = next(r for r in results if r["rule_id"] == "CREDIT_SCORE_MIN")
        assert credit_rule["violated"] is True

    def test_credit_above_threshold(self):
        facts = {"credit_score": 720}
        results = check_compliance(facts)
        credit_rule = next(r for r in results if r["rule_id"] == "CREDIT_SCORE_MIN")
        assert credit_rule["violated"] is False

    def test_dti_violation(self):
        facts = {"dti_ratio": 45.0}
        results = check_compliance(facts)
        dti_rule = next(r for r in results if r["rule_id"] == "DTI_RATIO")
        assert dti_rule["violated"] is True

    def test_bankruptcy_flag(self):
        facts = {"bankruptcy_mentioned": True}
        results = check_compliance(facts)
        bankruptcy_rule = next(r for r in results if r["rule_id"] == "BANKRUPTCY")
        assert bankruptcy_rule["violated"] is True

    def test_underage_applicant(self):
        facts = {"applicant_age": 17}
        results = check_compliance(facts)
        age_rule = next(r for r in results if r["rule_id"] == "APPLICANT_AGE")
        assert age_rule["violated"] is True
        assert age_rule["severity"] == "critical"

    def test_all_pass_good_applicant(self):
        facts = {
            "credit_score": 750,
            "dti_ratio": 25.0,
            "ltv_ratio": 70.0,
            "applicant_age": 35,
            "bankruptcy_mentioned": False,
            "first_time_buyer": False,
        }
        results = check_compliance(facts)
        violations = [r for r in results if r["violated"]]
        assert len(violations) == 0


class TestDetermineRecommendation:
    """Test the recommendation logic."""

    def test_approve_low_risk(self):
        rec = determine_recommendation(15, [])
        assert rec == "APPROVE"

    def test_deny_critical_violation(self):
        results = [{"violated": True, "severity": "critical"}]
        rec = determine_recommendation(30, results)
        assert rec == "DENY"

    def test_deny_very_high_score(self):
        rec = determine_recommendation(85, [])
        assert rec == "DENY"

    def test_manual_review_moderate_risk(self):
        rec = determine_recommendation(40, [])
        assert rec == "MANUAL_REVIEW"

    def test_manual_review_high_violations(self):
        results = [{"violated": True, "severity": "high"}]
        rec = determine_recommendation(20, results)
        assert rec == "MANUAL_REVIEW"


# ══════════════════════════════════════════════════════════════════
#  State / Trace Tests
# ══════════════════════════════════════════════════════════════════

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
