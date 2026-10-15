# Retrieval experiments (MLflow)

Run: `cd backend && python -m evaluation.run_eval --sweep`, then `mlflow ui --backend-store-uri sqlite:///mlflow.db` (experiment `guideline-retrieval`).
Golden set: 27 questions over HUD Handbook 4000.1 II.A.4. Metric: hit rate = an expected handbook page is in the top-k chunks. Measured 2026-10-15.

| Run | chunk / overlap | k | embeddings | chunks | hit rate | MRR | p50 ms | misses |
|---|---|---|---|---|---|---|---|---|
| **baseline (prod defaults)** | 1000 / 200 | 5 | openai text-embedding-3-small | 227 | 0.926 | 0.870 | 348 | 2 |
| | 500 / 100 | 5 | 3-small | 441 | 0.963 | 0.883 | 358 | 1 |
| | 750 / 150 | 5 | 3-small | 299 | 0.963 | 0.859 | 329 | 1 |
| | 1500 / 300 | 5 | 3-small | 174 | 0.963 | 0.849 | 345 | 1 |
| | 1000 / 200 | 3 | 3-small | 227 | 0.926 | 0.870 | 348 | 2 |
| | 1000 / 200 | 8 | 3-small | 227 | 0.926 | 0.870 | 346 | 2 |
| | 500 / 100 | 8 | 3-small | 441 | 1.000 | 0.887 | 380 | 0 |
| | 1000 / 200 | 5 | baai bge-base-en-v1.5 | 227 | 0.963 | 0.896 | 330 | 1 |
| **winner** | 500 / 100 | 5 | baai bge-base-en-v1.5 | 441 | **1.000** | 0.895 | 337 | 0 |
| | 1000 / 200 | 5 | openai text-embedding-3-large | 227 | 0.963 | 0.872 | 452 | 1 |

**Winner:** 500/100 chunks, k=5, bge-base: hit rate 92.6% → 100% (+7.4 pts, i.e. 2 more of 27 questions), MRR 0.870 → 0.895, latency unchanged (p50 337 vs 348 ms, within noise).
Smaller chunks are the main lever; k beyond 5 helps only with the weaker 1000-char chunking. A bigger embedding model (3-large) was not better and is slower.

**Caveats:** 27 questions means one question = 3.7 points, so differences of one question are not significant. The configuration was selected on the same questions it is scored on, so 100% is optimistic; a held-out split or more questions is needed before quoting it as generalization. No prompt-version axis here: this experiment is retrieval-only (no generation).
