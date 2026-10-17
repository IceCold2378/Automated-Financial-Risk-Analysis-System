"""Data loading and feature engineering for the credit-default risk model.

Dataset: "Default of Credit Card Clients" (Yeh & Lien, 2009), UCI Machine Learning Repository,
licensed CC BY 4.0, 30,000 Taiwanese credit-card clients. It is PUBLIC DATA and is not real
loan underwriting data. Downloaded on demand and cached under backend/data/risk/ (not committed).
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
CACHE = Path(__file__).resolve().parents[1] / "data" / "risk" / "credit_default.csv"
SEED = 42
TARGET = "default"

# SEX and MARRIAGE are protected attributes under ECOA and are never used as model inputs.
# They are kept in the frame only for the post-hoc group audit.
AUDIT_COLUMNS = ["SEX", "MARRIAGE"]
PAY = [f"PAY_{i}" for i in range(1, 7)]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_FEATURES = ["LIMIT_BAL", "AGE", "EDUCATION", *PAY, *BILL, *PAYAMT]
ENGINEERED = ["util_1", "max_delay", "n_late", "pay_ratio"]
FEATURES = RAW_FEATURES + ENGINEERED


def load_raw(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE)
    with urllib.request.urlopen(URL, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    df = pd.read_excel(z.open(z.namelist()[0]), header=1).drop(columns=["ID"])
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean codes and add engineered features. Works on training frames and API payloads."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})  # undocumented codes -> "other"
    df["util_1"] = (df["BILL_AMT1"] / df["LIMIT_BAL"].clip(lower=1)).clip(-1, 5)
    df["max_delay"] = df[PAY].max(axis=1)
    df["n_late"] = (df[PAY] > 0).sum(axis=1)
    df["pay_ratio"] = (df[PAYAMT].sum(axis=1) / df[BILL].clip(lower=0).sum(axis=1).clip(lower=1)).clip(0, 5)
    return df


def load_splits() -> dict:
    """Stratified 60/20/20 train/validation/test split with a fixed seed."""
    df = add_features(load_raw())
    train, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=SEED)
    return {"train": train, "val": val, "test": test}


def xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET].astype(int)


