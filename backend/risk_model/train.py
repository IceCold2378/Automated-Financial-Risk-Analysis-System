"""Train and compare a regularized logistic regression (GLM) and XGBoost; log to MLflow.

Run from backend/:  python -m risk_model.train
Model selection uses the validation set only; the test set is scored once at the end.
"""
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from risk_model.data import FEATURES, SEED, load_splits, xy  # noqa: E402
from risk_model.model import predict_proba, save_bundle  # noqa: E402

TRACKING_URI = "sqlite:///mlflow.db"
CATEGORICAL = ["EDUCATION"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


def ece(y, p, bins=10) -> float:
    """Expected calibration error with equal-frequency bins."""
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return float(np.mean(np.abs(frac - mean)))


def best_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.9, 86)
    return float(grid[int(np.argmax([f1_score(y, p >= t) for t in grid]))])


def logistic(C: float) -> Pipeline:
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=C, max_iter=3000, random_state=SEED))])


def xgboost(depth: int, X_val, y_val) -> XGBClassifier:
    return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=depth, min_child_weight=5,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                         early_stopping_rounds=30, random_state=SEED, n_jobs=2)


def metrics(y, p, thr) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, p >= thr).ravel()
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "ece": ece(y, p), "threshold": thr, "f1": f1_score(y, p >= thr),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    s = load_splits()
    Xtr, ytr = xy(s["train"]); Xva, yva = xy(s["val"]); Xte, yte = xy(s["test"])  # noqa: E702
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("risk-model")
    print(f"rows train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)}; default rate = {ytr.mean():.3f}")

    results = {}
    # ── Logistic regression: pick C on validation AUC ──
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        m = logistic(C).fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, m)
    # ── XGBoost: pick depth on validation AUC (early stopping on validation) ──
    best_x = None
    for depth in [3, 4, 6]:
        m = xgboost(depth, Xva, yva).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if best_x is None or auc > best_x[0]:
            best_x = (auc, depth, m)

    candidates = {"logistic_regression": (best[2], {"C": best[1]}), "xgboost": (best_x[2], {"max_depth": best_x[1],
                  "best_iteration": int(best_x[2].best_iteration)})}
    for name, (model, params) in candidates.items():
        bundle = {"model_name": name, "model": model, "params": params}
        pv, pt = predict_proba(bundle, s["val"]), predict_proba(bundle, s["test"])
        thr = best_threshold(yva, pv)  # threshold chosen on validation, applied to test
        m_val, m_test = roc_auc_score(yva, pv), metrics(yte, pt, thr)
        results[name] = {"bundle": bundle, "val_auc": m_val, "test": m_test, "p_test": pt, "threshold": thr}
        # fairness audit: test AUC by group (SEX is never a model input)
        audit = {}
        for col, groups in [("SEX", {1: "male", 2: "female"})]:
            for code, label in groups.items():
                mask = (s["test"][col] == code).values
                audit[f"auc_{col.lower()}_{label}"] = roc_auc_score(yte[mask], pt[mask])
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "seed": SEED, "n_features": len(FEATURES),
                               "excluded_protected": "SEX,MARRIAGE", **params})
            mlflow.log_metrics({"val_auc": m_val, **{f"test_{k}": v for k, v in m_test.items()}, **audit})
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                frac, mean = calibration_curve(yte, pt, n_bins=10, strategy="quantile")
                plt.figure(figsize=(4.5, 4.5))
                plt.plot([0, 1], [0, 1], "k--", label="perfect"); plt.plot(mean, frac, "o-", label=name)  # noqa: E702
                plt.xlabel("mean predicted probability"); plt.ylabel("observed default rate")  # noqa: E702
                plt.title(f"Reliability (test) ECE={m_test['ece']:.3f}"); plt.legend(); plt.tight_layout()  # noqa: E702
                plt.savefig(d / "reliability.png", dpi=120); plt.close()  # noqa: E702
                pd.DataFrame([[m_test["tn"], m_test["fp"]], [m_test["fn"], m_test["tp"]]],
                             index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(d / "confusion_matrix.csv")
                if name == "logistic_regression":
                    names = list(model.named_steps["prep"].get_feature_names_out())
                    coef = model.named_steps["clf"].coef_[0]
                    pd.DataFrame({"feature": names, "coef_per_sd": coef, "odds_ratio_per_sd": np.exp(coef)}
                                 ).sort_values("coef_per_sd", key=abs, ascending=False).to_csv(d / "odds_ratios.csv", index=False)
                mlflow.log_artifacts(str(d))
        print(f"{name:20s} val_auc={m_val:.4f} | test auc={m_test['auc']:.4f} pr_auc={m_test['pr_auc']:.4f} "
              f"brier={m_test['brier']:.4f} ece={m_test['ece']:.4f} thr={thr:.2f} f1={m_test['f1']:.3f} "
              f"cm(tn,fp,fn,tp)={m_test['tn']},{m_test['fp']},{m_test['fn']},{m_test['tp']} audit={ {k: round(v,3) for k,v in audit.items()} }")

    # Deploy the model with the higher VALIDATION auc.
    winner = max(results, key=lambda k: results[k]["val_auc"])
    r = results[winner]
    bundle = {**r["bundle"], "threshold": r["threshold"], "metrics": r["test"], "trained_at": time.strftime("%Y-%m-%d"),
              "dataset": "UCI Default of Credit Card Clients (CC BY 4.0), public data, not real underwriting data"}
    save_bundle(bundle)
    print("deployed:", winner)


if __name__ == "__main__":
    main()
