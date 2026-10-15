"""Retrieval evaluation against the golden question set.

A question is a *hit* when any of the top-k retrieved chunks comes from one of the
handbook pages listed in `expected_pages`. Shared by the pytest eval and the MLflow runner.
"""
import json
import time
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "tests" / "golden" / "questions.jsonl"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def evaluate_retrieval(retriever, questions: list[dict]) -> dict:
    """Run every question through `retriever`; return metrics and per-question rows."""
    rows, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        docs = retriever.invoke(q["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        pages = [d.metadata.get("page") for d in docs]
        expected = set(q["expected_pages"])
        rank = next((i + 1 for i, p in enumerate(pages) if p in expected), None)
        rows.append({
            "id": q["id"], "question": q["question"], "expected_pages": sorted(expected),
            "retrieved_pages": pages, "hit": rank is not None, "rank": rank,
        })
    n = len(rows)
    return {
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "n_questions": n,
        "rows": rows,
    }
