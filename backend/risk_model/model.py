"""Model bundle: scoring and per-prediction explanations for both model types."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from risk_model.data import FEATURES, add_features

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "risk_model.joblib"


def save_bundle(bundle: dict, path: Path = ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path = ARTIFACT) -> dict:
    return joblib.load(path)


def predict_proba(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(frame[FEATURES])[:, 1]


def contributions(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Per-feature contribution to the log-odds of default (rows x features).

    Logistic regression: coefficient * standardized value. XGBoost: built-in tree-SHAP (pred_contribs).
    Positive values push the predicted default risk up.
    """
    model = bundle["model"]
    if bundle["model_name"] == "logistic_regression":
        pre, clf = model.named_steps["prep"], model.named_steps["clf"]
        names = list(pre.get_feature_names_out())
        z = pre.transform(frame[FEATURES])
        z = z.toarray() if hasattr(z, "toarray") else np.asarray(z)
        raw = z * clf.coef_[0]
        # fold one-hot columns back onto their source feature
        out = np.zeros((len(frame), len(FEATURES)))
        for j, n in enumerate(names):
            src = n.split("__", 1)[1]
            base = next((f for f in FEATURES if src == f or src.startswith(f + "_")), None)
            out[:, FEATURES.index(base)] += raw[:, j]
        return out
    booster = model.get_booster()
    return booster.predict(xgb.DMatrix(frame[FEATURES]), pred_contribs=True)[:, :-1]


def top_factors(bundle: dict, payload: dict, n: int = 5) -> list[dict]:
    frame = add_features(pd.DataFrame([payload]))
    contrib = contributions(bundle, frame)[0]
    order = np.argsort(-np.abs(contrib))[:n]
    return [{"feature": FEATURES[i], "value": float(frame[FEATURES[i]].iloc[0]),
             "contribution": round(float(contrib[i]), 4),
             "effect": "increases risk" if contrib[i] > 0 else "decreases risk"} for i in order]


def risk_band(p: float, threshold: float) -> str:
    return "high" if p >= threshold else "elevated" if p >= threshold / 2 else "low"
