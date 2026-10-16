"""Unit tests for the MCP tool functions (offline: fake retriever and fake analyzer)."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from mcp_server import tools


class _Doc:
    def __init__(self, page, text="excerpt"):
        self.page_content, self.metadata = text, {"page": page, "section": "Evaluating Credit History"}


class _Retriever:
    def __init__(self):
        self.calls = []
        self.vectorstore = self

    def similarity_search(self, query, k):
        self.calls.append((query, k))
        return [_Doc(190 + i) for i in range(k)]


# ── search_guidelines ──
def test_search_returns_citations_and_respects_top_k():
    r = _Retriever()
    out = tools.search_guidelines("bankruptcy waiting period", 3, retriever=r)
    assert [o["page"] for o in out] == [190, 191, 192]
    assert out[0]["citation"] == "HUD Handbook 4000.1, p.190"
    assert r.calls == [("bankruptcy waiting period", 3)]


@pytest.mark.parametrize("query,top_k", [("", 5), ("   ", 5), ("x" * 501, 5), ("ok", 0), ("ok", 11)])
def test_search_validates_input(query, top_k):
    with pytest.raises(ValueError):
        tools.search_guidelines(query, top_k, retriever=_Retriever())


# ── get_citation ──
def test_get_citation_known_page():
    out = tools.get_citation(190)
    assert out["citation"] == "HUD Handbook 4000.1, p.190" and out["text"]


def test_get_citation_section_filter():
    out = tools.get_citation(190, "credit history")
    assert all("credit history" in s.lower() for s in out["sections"])
    with pytest.raises(ValueError, match="no section matching"):
        tools.get_citation(190, "nonexistent heading")


def test_get_citation_unknown_page():
    with pytest.raises(ValueError, match="not in the corpus"):
        tools.get_citation(5)


# ── list_documents / analyze_document ──
@pytest.fixture
def docs(tmp_path, monkeypatch):
    (tmp_path / "good_app.txt").write_text("Credit score: 700", encoding="utf-8")
    (tmp_path / "Bad Name.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")
    monkeypatch.setenv("MCP_DOCS_DIR", str(tmp_path))
    return tmp_path


def test_list_documents_only_safe_txt_ids(docs):
    assert [d["doc_id"] for d in tools.list_documents()] == ["good_app"]


def test_list_documents_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_DOCS_DIR", str(tmp_path / "nope"))
    assert tools.list_documents() == []


@pytest.mark.parametrize("bad", ["../secret", "..\\x", "good_app.txt", "GOOD", "", "a/b", "unknown_doc"])
def test_analyze_rejects_bad_or_unknown_ids(docs, bad):
    async def never(**kw):
        raise AssertionError("analyzer must not run")
    with pytest.raises(ValueError):
        asyncio.run(tools.analyze_document(bad, analyzer=never))


def test_analyze_document_shapes_result(docs):
    flag = SimpleNamespace(category="Credit", severity="high", description="d", confidence=0.9,
                           guideline_reference="HUD Handbook 4000.1, p.190 (S)")
    ungrounded = SimpleNamespace(category="Other", severity="low", description="d", confidence=0.4,
                                 guideline_reference="Uncited: no supporting handbook excerpt was retrieved")
    result = SimpleNamespace(summary="s", overall_risk_score=55, overall_risk_level="high",
                             recommendation="MANUAL_REVIEW", risk_flags=[flag, ungrounded],
                             processing_time_seconds=1.2)
    seen = {}

    async def analyzer(document_text, filename):
        seen.update(text=document_text, name=filename)
        return result, {}

    out = asyncio.run(tools.analyze_document("good_app", analyzer=analyzer))
    assert seen == {"text": "Credit score: 700", "name": "good_app.txt"}
    assert out["risk_score"] == 55 and out["grounded_flags"] == 1 and len(out["flags"]) == 2
    json.dumps(out)  # must be JSON-serializable for the MCP response


# ── server wiring ──
def test_server_registers_four_read_only_tools():
    from mcp_server import server
    registered = asyncio.run(server.mcp.list_tools())
    assert sorted(t.name for t in registered) == ["analyze_document", "get_citation",
                                                  "list_documents", "search_guidelines"]
    assert all(t.annotations.readOnlyHint for t in registered)
