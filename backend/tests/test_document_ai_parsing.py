"""Unit tests for document validation/extraction and LLM response parsing."""
import json

import pytest

from app.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.services.ai_service import _parse_llm_response
from app.services.document_service import extract_text, validate_file


def test_validate_rejects_unknown_extension():
    with pytest.raises(UnsupportedFileTypeError):
        validate_file("malware.exe", b"x")


def test_validate_rejects_oversize(monkeypatch):
    from app.services import document_service as ds
    monkeypatch.setattr(ds.settings, "MAX_FILE_SIZE_MB", 0)
    with pytest.raises(FileTooLargeError):
        validate_file("a.txt", b"x" * 10)


def test_extract_txt():
    assert "credit score 700" in extract_text("app.txt", b"credit score 700")


def test_parse_valid_json_clamps_values():
    raw = json.dumps({"summary": "ok", "overall_risk_score": 250, "recommendation": "APPROVE",
                      "risk_flags": [{"category": "Credit", "description": "d", "severity": "high",
                                      "guideline_reference": "p.188", "confidence": 3}]})
    r = _parse_llm_response(raw)
    assert r.overall_risk_score == 100
    assert r.risk_flags[0].confidence == 1.0
    assert r.risk_flags[0].guideline_reference == "p.188"


def test_parse_strips_markdown_fence():
    raw = "```json\n" + json.dumps({"summary": "fenced", "overall_risk_score": 10}) + "\n```"
    assert _parse_llm_response(raw).summary == "fenced"


def test_parse_invalid_json_falls_back_to_manual_review():
    r = _parse_llm_response("not json at all")
    assert r.recommendation == "MANUAL_REVIEW"
    assert r.overall_risk_score == 50


# ── Document extraction on the bundled samples ──
from pathlib import Path  # noqa: E402

from app.exceptions import AIServiceError, DocumentParsingError  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def test_extract_sample_pdf():
    assert len(extract_text("sample_app.pdf", (DATA / "sample_app.pdf").read_bytes()).strip()) > 50


def test_extract_sample_docx():
    assert len(extract_text("tester.docx", (DATA / "tester.docx").read_bytes()).strip()) > 0


def test_empty_txt_rejected():
    with pytest.raises(DocumentParsingError):
        extract_text("e.txt", b"   ")


def test_latin1_txt_fallback():
    assert "caf" in extract_text("l.txt", "café".encode("latin-1"))


def test_corrupt_pdf_rejected():
    with pytest.raises(DocumentParsingError):
        extract_text("bad.pdf", b"not a pdf")


# ── analyze_document: agent path, fallback path, error path (no LLM calls) ──
async def test_analyze_document_uses_agent_result(monkeypatch):
    from app.agent import orchestrator
    from app.models.schemas import AnalysisResult
    from app.services import ai_service

    expected = AnalysisResult(summary="agent", overall_risk_score=12, overall_risk_level="low",
                              recommendation="APPROVE", detailed_analysis="d")

    async def fake_agent(**kwargs):
        return expected, {"steps": []}

    monkeypatch.setattr(orchestrator, "run_agent_analysis", fake_agent)
    monkeypatch.setattr(ai_service.settings, "AGENT_ENABLE_TOOLS", True)
    assert (await ai_service.analyze_document("text")).summary == "agent"


async def test_analyze_document_falls_back_to_rag_chain(monkeypatch):
    from app.agent import orchestrator
    from app.services import ai_service

    async def boom(**kwargs):
        raise RuntimeError("agent down")

    class FakeChain:
        def invoke(self, text):
            return json.dumps({"summary": "rag", "overall_risk_score": 33})

    monkeypatch.setattr(orchestrator, "run_agent_analysis", boom)
    monkeypatch.setattr(ai_service.settings, "AGENT_ENABLE_TOOLS", True)
    monkeypatch.setattr(ai_service, "_rag_chain", FakeChain())
    r = await ai_service.analyze_document("text")
    assert r.summary == "rag" and r.overall_risk_score == 33


async def test_analyze_document_wraps_chain_errors(monkeypatch):
    from app.services import ai_service

    class BadChain:
        def invoke(self, text):
            raise RuntimeError("llm timeout")

    monkeypatch.setattr(ai_service.settings, "AGENT_ENABLE_TOOLS", False)
    monkeypatch.setattr(ai_service, "_rag_chain", BadChain())
    with pytest.raises(AIServiceError):
        await ai_service.analyze_document("text")
