# Baseline (measured 2026-10-15, local machine, OpenRouter)

Models: chat `google/gemini-2.5-flash`, embeddings `openai/text-embedding-3-small` via OpenRouter.

| Metric | Value | How measured |
|---|---|---|
| Tests | 74 → 99 (+1 eval) | `pytest` |
| Line coverage (app) | 53% → 58% | `pytest --cov=app` (same code, old vs. new test set) |
| Retrieval hit@5 on golden set | **92.6%** (25/27) | `pytest -m eval -s`; chunk 1000 / overlap 200 / k=5 |
| Retrieval MRR | 0.870 | same run |
| Retrieval latency (query embedding + search) | p50 352 ms (eval run), 410 ms (earlier run) | same |
| End-to-end analysis latency | mean 4.18 s, p50 4.46 s, p95 5.45 s, 0/20 errors | 20 in-process `analyze_document` calls cycling over 3 sample applications; includes LLM call |

Golden set: 27 questions, each tied to the HUD Handbook 4000.1 printed page(s) that answer it; an offline test checks that every expected page contains a hand-chosen evidence phrase. Hit = an expected page appears in the top-5 chunks. Misses: q04 (foreclosure waiting period), q10 (income continuance).

Not yet measured: end-to-end citation accuracy of the final analysis (blocked by the known issue that citations come from hard-coded rules), docker-compose setup time.
