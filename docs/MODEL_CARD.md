# Model card: credit-default risk score (`POST /api/v1/risk/score`)

**Public-data demo, not real underwriting.** The model is trained on the UCI "Default of Credit Card Clients" dataset (Yeh & Lien, 2009; 30,000 Taiwanese credit-card clients, April-September 2005; CC BY 4.0). It predicts next-month credit-card default. It was not built on loan applications, must not drive real credit decisions, and has not been validated for any population other than the one in the data.

## Data and split
- 30,000 rows, 22.1% default rate. Stratified 60/20/20 train/validation/test split, `random_state=42` (18,000 / 6,000 / 6,000).
- Inputs: credit limit, age, education, 6 months each of repayment status, bill amount and payment amount, plus four engineered features (`util_1`, `max_delay`, `n_late`, `pay_ratio`; see `risk_model/data.py`).
- **Protected attributes excluded:** `SEX` and `MARRIAGE` are never model inputs (ECOA). Sex is kept only to audit the test-set error rates by group.
- Model choice, hyperparameters and decision threshold were selected on the validation set only. The test set was scored once.

## Models and results (held-out test set, MLflow experiment `risk-model`)
| Model | AUC | PR-AUC | Brier | ECE (10 bins) | F1 @ threshold |
|---|---|---|---|---|---|
| Logistic regression (L2 GLM, C tuned) | 0.758 | 0.520 | 0.139 | 0.015 | 0.540 @ 0.28 |
| **XGBoost (deployed)** | **0.792** | **0.583** | **0.131** | **0.010** | 0.556 @ 0.25 |

Confusion matrix at the validation-chosen threshold (test, tn / fp / fn / tp): logistic 3935 / 738 / 564 / 763; XGBoost 3874 / 799 / 509 / 818. Accuracy is a poor headline here because 78% of clients do not default, so I report AUC, PR-AUC and F1.

**Calibration:** both models are well calibrated on the test set (ECE 0.015 and 0.010; reliability curves in the MLflow artifacts `reliability.png`), so predicted probabilities can be read as probabilities for this population without an extra calibration step. XGBoost is the deployed model because it had the higher validation AUC (0.779 vs 0.739).

**GLM interpretation** (odds ratios per +1 SD, `odds_ratios.csv`): the number of late months (`n_late`) is the strongest driver (OR 2.13), then worst delay (`max_delay`, OR 1.35) and the latest repayment status (`PAY_1`, OR 1.26); higher credit limit (OR 0.84) and larger recent payments (OR 0.80) lower the odds. `EDUCATION=other` shows a large protective effect (OR 0.28); that category is small and probably reflects data coding, so do not read it causally.

**Fairness audit (test AUC by sex, sex not used as an input):** logistic 0.759 male / 0.755 female; XGBoost 0.782 male / 0.798 female. Similar discrimination, which is not the same as equal approval rates or equal error rates; those were not evaluated.

## Endpoint
`POST /api/v1/risk/score` (JWT required) takes the 21 repayment fields and returns the probability of default, a 0-100 score, a risk band (`high` at or above the model threshold, `elevated` at half of it), and the top 5 contributing features (tree-SHAP contributions in log-odds for XGBoost). It returns 503 if the artifact is missing; retrain with `cd backend && python -m risk_model.train`.

## Limitations
- Single dataset, single country and period (2005); no temporal validation; the dataset's own labeling and coding quirks (undocumented education codes, repayment-status coding) carry through.
- Test AUC has sampling error; with 6,000 test rows and 1,327 defaults, treat differences of about 0.01 as noise. The logistic-vs-XGBoost gap (0.034) is larger than that but was measured on one split, not repeated cross-validation.
- Not tuned heavily (3 depths, 4 regularization strengths). No monotonic constraints, reject inference or cost-sensitive threshold.
- Age is used as an input; that is permitted only under specific conditions in US credit rules and should be reviewed before any real use.
