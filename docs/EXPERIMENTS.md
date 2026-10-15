# Retrieval experiments (MLflow)

Run: `cd backend && python -m evaluation.run_eval --sweep`, then `mlflow ui --backend-store-uri sqlite:///mlflow.db` (experiment `guideline-retrieval`).
Golden set: 27 questions over HUD Handbook 4000.1 II.A.4. Hit rate = an expected handbook page is in the top-k chunks. Sweep re-run 2026-10-15 (14 configurations; latency is query embedding + search, via OpenRouter).

| chunk / overlap | k | embeddings | chunks | hit rate | MRR | p50 ms | misses |
|---|---|---|---|---|---|---|---|
| **1000 / 200 (old default)** | 5 | openai text-embedding-3-small | 227 | 0.926 | 0.870 | 329 | 2 |
| 500 / 100 | 5 | 3-small | 441 | 0.963 | 0.864 | 336 | 1 |
| 750 / 150 | 5 | 3-small | 299 | 0.963 | 0.859 | 361 | 1 |
| 1500 / 300 | 5 | 3-small | 174 | 0.963 | 0.849 | 337 | 1 |
| 1000 / 200 | 3 | 3-small | 227 | 0.926 | 0.870 | 374 | 2 |
| 1000 / 200 | 8 | 3-small | 227 | 0.926 | 0.870 | 321 | 2 |
| 500 / 100 | 8 | 3-small | 441 | 1.000 | 0.887 | 355 | 0 |
| 1000 / 200 | 5 | baai bge-base-en-v1.5 | 227 | 0.963 | 0.896 | 340 | 1 |
| 500 / 100 | 5 | baai bge-base-en-v1.5 | 441 | 1.000 | 0.895 | 315 | 0 |
| 1000 / 200 | 5 | openai text-embedding-3-large | 227 | 0.963 | 0.872 | 408 | 1 |
| 500 / 100 | 5 | openai text-embedding-3-large | 441 | 0.926 | 0.889 | 395 | 2 |
| 500 / 100 | 5 | qwen3-embedding-8b | 441 | 0.963 | 0.864 | 1081 | 1 |
| 500 / 100 | 5 | google gemini-embedding-001 | n/a | failed: OpenRouter rejected the batch request (400) | | | |
| **500 / 100 (new default)** | 5 | **baai bge-large-en-v1.5** | 441 | **1.000** | **0.910** | 401 | 0 |

**Chosen configuration:** 500/100 chunks, k=5, `baai/bge-large-en-v1.5`. Hit rate 92.6% → 100% (+7.4 pts, 2 more of 27 questions), MRR 0.870 → 0.910. Retrieval latency rose from ~330 ms to ~400 ms p50 (the larger embedding model); bge-base matches the hit rate at ~315 ms if latency matters more than rank quality.
Bigger is not automatically better: OpenAI 3-large was worse than 3-small at 500/100 and slower; qwen3-8b was 3x slower with no gain. Chunk size was the main lever; k only mattered with the larger chunks.

**Caveats:** 27 questions means one question = 3.7 points, so single-question differences (e.g. bge-base vs bge-large on hit rate) are not significant. The configuration was selected on the same questions it is scored on, so 100% is optimistic; a held-out split or more questions is needed before quoting it as generalization. This experiment is retrieval-only (no prompt-version axis).
