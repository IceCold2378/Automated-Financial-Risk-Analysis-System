"""Golden-set integrity (offline) and retrieval quality (eval, needs OPENROUTER_API_KEY)."""
import json
import os
import re

import pytest

from app.config import get_settings
from evaluation.retrieval import evaluate_retrieval, load_golden, percentile

CORPUS = get_settings().effective_guidelines_path


def _pages():
    pages = {}
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pages[r["page"]] = pages.get(r["page"], "") + " " + r["text"]
    return pages


def test_golden_set_size_and_unique_ids():
    g = load_golden()
    assert 20 <= len(g) <= 40
    assert len({q["id"] for q in g}) == len(g)


def test_every_golden_citation_is_supported_by_corpus():
    """Each expected page must exist and contain the hand-checked evidence phrase."""
    pages = _pages()
    for q in load_golden():
        for p in q["expected_pages"]:
            text = re.sub(r"\s+", " ", pages.get(p, ""))
            assert re.search(q["evidence"], text, re.I), f"{q['id']}: evidence missing on page {p}"


class _Doc:
    def __init__(self, page):
        self.metadata = {"page": page}


class _FakeRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def invoke(self, question):
        return [_Doc(p) for p in self.mapping.get(question, [])]


def test_evaluate_retrieval_metrics():
    qs = [{"id": "a", "question": "qa", "expected_pages": [1]},
          {"id": "b", "question": "qb", "expected_pages": [2]},
          {"id": "c", "question": "qc", "expected_pages": [3]}]
    r = _FakeRetriever({"qa": [1, 9], "qb": [9, 2], "qc": [8, 9]})
    m = evaluate_retrieval(r, qs)
    assert m["hit_rate"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert [row["hit"] for row in m["rows"]] == [True, True, False]


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([7], 0.95) == 7


@pytest.mark.eval
@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY") and not get_settings().OPENROUTER_API_KEY,
                    reason="needs OPENROUTER_API_KEY for embeddings")
def test_retrieval_hit_rate_on_golden_set(tmp_path):
    from app.services.vector_service import build_vector_db
    retriever = build_vector_db(persist_dir=str(tmp_path / "chroma"), collection_name="eval_test")
    m = evaluate_retrieval(retriever, load_golden())
    print(f"\nhit_rate={m['hit_rate']:.3f} mrr={m['mrr']:.3f} p50={m['latency_p50_ms']:.0f}ms")
    assert m["hit_rate"] >= 0.85
