"""Unit tests for the vector service and corpus (offline: fake embeddings, no network)."""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.services import vector_service as vs


@pytest.fixture
def jsonl_corpus(tmp_path):
    p = tmp_path / "corpus.jsonl"
    recs = [{"page": i, "section": f"S{i}", "text": f"Rule {i}. " + "word " * 120} for i in range(1, 4)]
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    return str(p)


def test_load_jsonl_keeps_page_metadata(jsonl_corpus):
    docs = vs.load_guideline_documents(jsonl_corpus)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]
    assert docs[0].metadata["section"] == "S1"


def test_load_txt(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("Minimum credit score is 680.", encoding="utf-8")
    assert "680" in vs.load_guideline_documents(str(p))[0].page_content


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vs.load_guideline_documents(str(tmp_path / "nope.jsonl"))


def test_split_inherits_metadata_and_respects_size():
    doc = Document(page_content="alpha beta gamma. " * 100, metadata={"page": 7})
    chunks = vs.split_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.metadata["page"] == 7 and len(c.page_content) <= 200 for c in chunks)


def test_smaller_chunks_yield_more_chunks():
    doc = Document(page_content="sentence about credit. " * 200)
    big = vs.split_documents([doc], 1000, 100)
    small = vs.split_documents([doc], 250, 25)
    assert len(small) > len(big)


def test_build_vector_db_reuses_persisted_collection(jsonl_corpus, tmp_path):
    kw = dict(guidelines_path=jsonl_corpus, embeddings=DeterministicFakeEmbedding(size=16),
              persist_dir=str(tmp_path / "chroma"), collection_name="test_coll", chunk_size=200, chunk_overlap=0, k=2)
    r1 = vs.build_vector_db(**kw)
    assert len(r1.invoke("rule")) == 2
    count = r1.vectorstore._collection.count()
    r2 = vs.build_vector_db(**kw)  # second start must not duplicate chunks
    assert r2.vectorstore._collection.count() == count
    r3 = vs.build_vector_db(force_rebuild=True, **kw)
    assert r3.vectorstore._collection.count() == count


def test_get_retriever_returns_cached(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(vs, "_retriever", sentinel)
    assert vs.get_retriever() is sentinel
