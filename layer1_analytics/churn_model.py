"""Piece 3: early-warning (churn) model with explanations.

Usage (from the repo root):
    python -m layer1_analytics.churn_model --demo
    python -m layer1_analytics.churn_model --csv data/raw/gym_churn.csv --target Churn --drop member_id

Uses LightGBM + SHAP when installed, otherwise falls back to scikit-learn's
HistGradientBoostingClassifier + permutation importance so the pipeline still runs.
--demo generates SYNTHETIC data to test the pipeline; never report its numbers as findings.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import config

POSITIVE = {"1", "yes", "y", "true", "t", "churn", "churned", "left"}
NEGATIVE = {"0", "no", "n", "false", "f", "stay", "stayed", "retained", "active"}


def demo_dataset(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 65, n),
        "tenure_months": rng.integers(1, 48, n),
        "visits_per_week": np.clip(rng.normal(2.5, 1.5, n), 0, 7).round(1),
        "avg_session_min": np.clip(rng.normal(55, 15, n), 15, 120).round(0),
        "group_class_rate": rng.uniform(0, 1, n).round(2),
        "has_trainer": rng.integers(0, 2, n),
        "contract": rng.choice(["monthly", "annual"], n, p=[0.6, 0.4]),
    })
    z = (2.6 - 0.8 * df["visits_per_week"] - 0.03 * df["tenure_months"]
         - 0.6 * df["has_trainer"] - 0.8 * (df["contract"] == "annual") + rng.normal(0, 0.8, n))
    df["churn"] = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    return df


def to_binary(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    m = s.astype(str).str.strip().str.lower()
    if not set(m.unique()) <= (POSITIVE | NEGATIVE):
        raise ValueError(f"Cannot map target values to 0/1: {sorted(m.unique())[:10]}")
    return m.isin(POSITIVE).astype(int)


def make_model(seed: int):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                  random_state=seed, verbose=-1), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed), "sklearn-hgb"


def explain(model, X_test, y_test, seed: int):
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        imp = pd.DataFrame({"feature": X_test.columns, "importance": np.abs(sv).mean(axis=0)})
        return imp.sort_values("importance", ascending=False), "mean |SHAP|"
    except Exception:
        r = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                   n_repeats=5, random_state=seed)
        imp = pd.DataFrame({"feature": X_test.columns, "importance": r.importances_mean})
        return imp.sort_values("importance", ascending=False), "permutation importance (AUC drop)"


def run(df: pd.DataFrame, target: str, drop=(), seed: int = 42, out_dir=None, label="data"):
    out_dir = out_dir or (config.OUTPUT_DIR / "churn")
    out_dir.mkdir(parents=True, exist_ok=True)
    y = to_binary(df[target])
    X = pd.get_dummies(df.drop(columns=[target, *drop]), drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    model, kind = make_model(seed)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "data": label, "model": kind, "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "churn_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
    }
    imp, how = explain(model, X_te, y_te, seed)
    metrics["importance_method"] = how
    imp.to_csv(out_dir / "feature_importance.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"model": model, "columns": list(X.columns)}, out_dir / "model.joblib")
    return metrics, imp


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--csv", default=str(config.GYM_CHURN_FILE))
    p.add_argument("--target", default="Churn", help="name of the churn column")
    p.add_argument("--drop", nargs="*", default=[], help="columns to drop (IDs etc.)")
    p.add_argument("--demo", action="store_true", help="run on synthetic data")
    a = p.parse_args()
    if a.demo:
        df, target, label, drop = demo_dataset(), "churn", "SYNTHETIC demo", []
    else:
        df, target, label, drop = pd.read_csv(a.csv), a.target, "kaggle gym churn", a.drop
    metrics, imp = run(df, target, drop, label=label)
    print(json.dumps(metrics, indent=2))
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
