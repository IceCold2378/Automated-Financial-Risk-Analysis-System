"""Run the golden-set retrieval eval for one or many configurations and log to MLflow.

Examples (from backend/):
    python -m evaluation.run_eval --chunk-size 500 --chunk-overlap 100 --k 5
    python -m evaluation.run_eval --sweep            # preset grid of configurations
    mlflow ui --backend-store-uri sqlite:///mlflow.db  # compare runs
Every number is computed from a live run; nothing is hard-coded.
"""
import argparse
import csv
import tempfile
import uuid
from pathlib import Path

import mlflow

from app.config import get_settings
from app.services import vector_service as vs
from evaluation.retrieval import evaluate_retrieval, load_golden

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "guideline-retrieval"
DEFAULT_EMB = "openai/text-embedding-3-small"

# (chunk_size, chunk_overlap, k, embedding_model); first entry is the production baseline.
SWEEP = [
    (1000, 200, 5, DEFAULT_EMB),
    (500, 100, 5, DEFAULT_EMB),
    (750, 150, 5, DEFAULT_EMB),
    (1500, 300, 5, DEFAULT_EMB),
    (1000, 200, 3, DEFAULT_EMB),
    (1000, 200, 8, DEFAULT_EMB),
    (500, 100, 8, DEFAULT_EMB),
    (1000, 200, 5, "baai/bge-base-en-v1.5"),
    (500, 100, 5, "baai/bge-base-en-v1.5"),
    (1000, 200, 5, "openai/text-embedding-3-large"),
    (500, 100, 5, "baai/bge-large-en-v1.5"),
    (500, 100, 5, "qwen/qwen3-embedding-8b"),
    (500, 100, 5, "google/gemini-embedding-001"),
    (500, 100, 5, "openai/text-embedding-3-large"),
]


def run_one(chunk_size: int, chunk_overlap: int, k: int, embedding_model: str, tag: str = "") -> dict:
    settings = get_settings()
    settings.EMBEDDING_MODEL = embedding_model
    questions = load_golden()
    name = f"cs{chunk_size}-ov{chunk_overlap}-k{k}-{embedding_model.split('/')[-1]}"
    with mlflow.start_run(run_name=name):
        params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "top_k": k,
                  "embedding_model": embedding_model, "corpus": Path(settings.effective_guidelines_path).name,
                  "n_questions": len(questions)}
        mlflow.log_params(params)
        if tag:
            mlflow.set_tag("note", tag)
        retriever = vs.build_vector_db(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k,
            embeddings=vs.get_embeddings(), persist_dir="", collection_name=f"eval_{uuid.uuid4().hex[:12]}",
        )
        n_chunks = retriever.vectorstore._collection.count()
        m = evaluate_retrieval(retriever, questions)
        mlflow.log_metrics({"hit_rate": m["hit_rate"], "mrr": m["mrr"], "latency_p50_ms": m["latency_p50_ms"],
                            "latency_p95_ms": m["latency_p95_ms"], "n_chunks": n_chunks})
        failed = [r for r in m["rows"] if not r["hit"]]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "failed_cases.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "question", "expected_pages", "retrieved_pages"])
                for r in failed:
                    w.writerow([r["id"], r["question"], r["expected_pages"], r["retrieved_pages"]])
            mlflow.log_artifact(str(path))
        print(f"{name}: hit_rate={m['hit_rate']:.3f} mrr={m['mrr']:.3f} "
              f"p50={m['latency_p50_ms']:.0f}ms chunks={n_chunks} failed={len(failed)}")
        return {"name": name, **{k_: m[k_] for k_ in ("hit_rate", "mrr", "latency_p50_ms", "latency_p95_ms")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embedding-model", default=DEFAULT_EMB)
    ap.add_argument("--sweep", action="store_true", help="run the preset configuration grid")
    ap.add_argument("--tracking-uri", default=TRACKING_URI)
    a = ap.parse_args()
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    configs = SWEEP if a.sweep else [(a.chunk_size, a.chunk_overlap, a.k, a.embedding_model)]
    for cfg in configs:
        try:
            run_one(*cfg)
        except Exception as e:  # keep the sweep going if one embedding model is unavailable
            print(f"FAILED {cfg}: {str(e)[:150]}")


if __name__ == "__main__":
    main()
