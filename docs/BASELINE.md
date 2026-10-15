# Baseline and results (measured 2026-10-15, local machine, OpenRouter)

Chat model `google/gemini-2.5-flash`. "Before" = repo state at the start of this work; "after" = branch `feat/hud-citations`.

| Metric | Before | After | How measured |
|---|---|---|---|
| Tests | 74 | 105 (+1 eval) | `pytest` |
| Line coverage (app) | 53% | 59% | `pytest --cov=app`; agent module ~65% in the final run |
| Guideline corpus | 8 rules, 1 chunk | 92 handbook pages, 441 chunks | HUD Handbook 4000.1 II.A.4 |
| Retrieval hit rate (golden set, 27 q) | 92.6% (1000/200, 3-small) | 100% (500/100, bge-large) | `pytest -m eval -s`, `evaluation.run_eval` |
| Retrieval MRR | 0.870 | 0.910 | same |
| Page-level citations in the final analysis | none (citations were hard-coded rule text) | every flag cites a retrieved handbook page | `evaluation.e2e` |
| Grounded-citation rate (flags citing a retrieved page) | n/a | 100%, 100%, 100% (3 runs; 93.5% in one earlier run with the bge-base config) | `evaluation.e2e` |
| End-to-end citation hit rate (10 planted-issue applications) | n/a | 70%, 80%, 70% (3 runs) | `evaluation.e2e` |
| End-to-end latency per analysis | 4.2 s mean, 5.5 s p95 (20 runs, old pipeline) | 4.7-5.8 s mean, 6.9-8.6 s p95 (10 runs each) | wall clock incl. LLM |

**What the e2e metric means:** 10 synthetic applications each plant one issue answered by specific handbook pages (`tests/golden/applications.jsonl`). A hit means a grounded flag cites an expected page. The control application (no issues) was approved.
**Caveats:** the application set is small and I widened a few `expected_pages` after the first run, adding equivalent pages from the manual-underwriting section that I verified state the same rule (see git history). LLM output varies run to run (temperature 0.1), hence the range. "Before" has no comparable citation number because the old system produced no page citations.
