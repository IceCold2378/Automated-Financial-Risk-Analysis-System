"""End-to-end citation evaluation: run synthetic applications through the full agent.

Each application in tests/golden/applications.jsonl plants one issue that a specific HUD
Handbook 4000.1 page answers. Metrics (all computed from the live run):
  e2e_hit_rate    - an expected page is cited by at least one *grounded* flag
  grounded_rate   - share of all flags whose citation points at a retrieved page
  cited_flag_rate - share of all flags carrying a page citation at all
  latency         - wall-clock seconds per analysis

Usage (from backend/):  python -m evaluation.e2e [--mlflow]
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from evaluation.retrieval import percentile

APPS_PATH = Path(__file__).resolve().parents[1] / "tests" / "golden" / "applications.jsonl"
_PAGE = re.compile(r"p\.(\d+)")


def load_apps(path: Path = APPS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def evaluate_agent(apps: list[dict]) -> dict:
    from app.agent.orchestrator import run_agent_analysis

    rows, lat = [], []
    for a in apps:
        t0 = time.perf_counter()
        result, trace = await run_agent_analysis(document_text=a["text"], filename=f"{a['id']}.txt")
        lat.append(time.perf_counter() - t0)
        flags = result.risk_flags
        cited = [int(m.group(1)) for f in flags for m in [_PAGE.search(f.guideline_reference)] if m]
        grounded_flags = sum(1 for f in flags if not f.guideline_reference.startswith("Uncited"))
        expected = set(a["expected_pages"])
        rows.append({"id": a["id"], "expected": sorted(expected), "cited_pages": cited, "flags": len(flags),
                     "grounded_flags": grounded_flags, "hit": bool(expected & set(cited)),
                     "recommendation": result.recommendation, "score": result.overall_risk_score})
    scored = [r for r in rows if r["expected"]]
    total_flags = sum(r["flags"] for r in rows)
    return {
        "e2e_hit_rate": sum(r["hit"] for r in scored) / len(scored),
        "grounded_rate": (sum(r["grounded_flags"] for r in rows) / total_flags) if total_flags else 1.0,
        "cited_flag_rate": (sum(len(r["cited_pages"]) for r in rows) / total_flags) if total_flags else 1.0,
        "latency_mean_s": sum(lat) / len(lat),
        "latency_p95_s": percentile(lat, 0.95),
        "n_apps": len(scored), "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()

    from app.services.ai_service import build_rag_chain
    from app.services.vector_service import build_vector_db
    build_rag_chain(build_vector_db())

    m = asyncio.run(evaluate_agent(load_apps()))
    for r in m["rows"]:
        print(r)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items() if k != "rows"})
    if a.mlflow:
        import mlflow
        from app.config import get_settings
        s = get_settings()
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("agent-citations-e2e")
        with mlflow.start_run(run_name="hud-grounded-citations"):
            mlflow.log_params({"chunk_size": s.CHUNK_SIZE, "chunk_overlap": s.CHUNK_OVERLAP, "top_k": s.VECTOR_SEARCH_K,
                               "embedding_model": s.EMBEDDING_MODEL, "llm": s.OPENROUTER_MODEL, "n_apps": m["n_apps"]})
            mlflow.log_metrics({k: v for k, v in m.items() if k not in ("rows", "n_apps")})


if __name__ == "__main__":
    main()
