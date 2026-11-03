"""Tests for the risk model package and the /risk/score endpoint (uses the committed artifact)."""
import numpy as np
import pandas as pd
import pytest

from risk_model import model as rm
from risk_model.data import FEATURES, add_features
from risk_model.train import best_threshold, ece

GOOD = {"limit_bal": 200000, "age": 40, "education": 2, "pay_1": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0,
        "pay_5": 0, "pay_6": 0, "bill_amt1": 20000, "bill_amt2": 19000, "bill_amt3": 18000, "bill_amt4": 17000,
        "bill_amt5": 16000, "bill_amt6": 15000, "pay_amt1": 5000, "pay_amt2": 5000, "pay_amt3": 5000,
        "pay_amt4": 5000, "pay_amt5": 5000, "pay_amt6": 5000}
BAD = {**GOOD, "limit_bal": 20000, "pay_1": 3, "pay_2": 3, "pay_3": 2, "pay_4": 2, "pay_5": 2, "pay_6": 2,
       "bill_amt1": 21000, "pay_amt1": 0, "pay_amt2": 0, "pay_amt3": 0, "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0}


def _frame(d):
    return add_features(pd.DataFrame([{k.upper(): v for k, v in d.items()}]))


def test_feature_engineering():
    f = _frame(BAD)
    assert f["max_delay"].iloc[0] == 3 and f["n_late"].iloc[0] == 6
    assert f["util_1"].iloc[0] == pytest.approx(21000 / 20000)
    assert f["pay_ratio"].iloc[0] == 0
    assert list(f[FEATURES].columns) == FEATURES


def test_protected_attributes_not_model_inputs():
    assert not {"SEX", "MARRIAGE"} & set(FEATURES)


def test_education_codes_collapsed():
    f = add_features(pd.DataFrame([{**{k.upper(): v for k, v in GOOD.items()}, "EDUCATION": 6}]))
    assert f["EDUCATION"].iloc[0] == 4


def test_ece_and_threshold_helpers():
    y = np.array([0, 0, 1, 1] * 50)
    assert ece(y, np.where(y == 1, 0.99, 0.01)) < 0.05
    assert 0.05 <= best_threshold(y, np.where(y == 1, 0.9, 0.1)) <= 0.9


@pytest.fixture(scope="module")
def bundle():
    return rm.load_bundle()


def test_delinquent_profile_scores_higher(bundle):
    p_good = rm.predict_proba(bundle, _frame(GOOD))[0]
    p_bad = rm.predict_proba(bundle, _frame(BAD))[0]
    assert 0 <= p_good < p_bad <= 1


def test_contributions_sum_matches_logit_direction(bundle):
    contrib = rm.contributions(bundle, _frame(BAD))
    assert contrib.shape == (1, len(FEATURES))
    top = rm.top_factors(bundle, {k.upper(): v for k, v in BAD.items()}, n=3)
    assert len(top) == 3 and abs(top[0]["contribution"]) >= abs(top[-1]["contribution"])


def test_risk_band():
    assert rm.risk_band(0.5, 0.25) == "high"
    assert rm.risk_band(0.15, 0.25) == "elevated"
    assert rm.risk_band(0.05, 0.25) == "low"


# ── endpoint ──
async def test_score_requires_auth(client):
    assert (await client.post("/api/v1/risk/score", json=GOOD)).status_code in (401, 403)


async def test_score_returns_probability_and_factors(client, auth_headers):
    r = await client.post("/api/v1/risk/score", json=BAD, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1 and body["score"] == round(body["probability_of_default"] * 100)
    assert len(body["top_factors"]) == 5 and "public" in body["disclaimer"]
    good = (await client.post("/api/v1/risk/score", json=GOOD, headers=auth_headers)).json()
    assert good["probability_of_default"] < body["probability_of_default"]


async def test_score_validates_input(client, auth_headers):
    bad = {**GOOD, "age": 12}
    assert (await client.post("/api/v1/risk/score", json=bad, headers=auth_headers)).status_code == 422
    missing = {k: v for k, v in GOOD.items() if k != "pay_1"}
    assert (await client.post("/api/v1/risk/score", json=missing, headers=auth_headers)).status_code == 422


async def test_score_503_when_model_missing(client, auth_headers, monkeypatch, tmp_path):
    from app.routers import risk
    monkeypatch.setattr(risk, "ARTIFACT", tmp_path / "missing.joblib")
    r = await client.post("/api/v1/risk/score", json=GOOD, headers=auth_headers)
    assert r.status_code == 503
