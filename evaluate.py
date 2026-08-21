"""Leakage-safe 5-fold evaluation and experiment tracking for LOQQIN."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVR

from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "questions.csv"
EXPERIMENT_DIR = ROOT / "experiments"
RANDOM_STATE = 42
N_SPLITS = 5


def tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english", ngram_range=(1, 3), max_features=5000,
        lowercase=True, token_pattern=r"(?u)\b[a-zA-Z]+\b", min_df=1,
        sublinear_tf=True,
    )


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).dropna(subset=["question", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise")
    if not df["label"].between(0, 10).all():
        raise ValueError("Labels must be between 0 and 10")
    df["text"] = df["question"].map(clean_text)
    return df[df["text"].str.strip().ne("")].reset_index(drop=True)


def regression_models() -> dict:
    return {
        "Ridge": Pipeline([("tfidf", tfidf()), ("model", Ridge(alpha=1.0))]),
        "GradientBoosting": Pipeline([("tfidf", tfidf()), ("model", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE))]),
        "SVR": Pipeline([("tfidf", tfidf()), ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))]),
        "StackingRegressor": Pipeline([
            ("tfidf", tfidf()),
            ("model", StackingRegressor(
                estimators=[
                    ("ridge", Ridge(alpha=1.0)),
                    ("gbr", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)),
                    ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
                ],
                final_estimator=Ridge(alpha=0.5), cv=3, n_jobs=-1,
            )),
        ]),
    }


def classification_models() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("tfidf", tfidf()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", tfidf()),
            ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def summarize(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}


def evaluate_regression(X: pd.Series, y: np.ndarray) -> dict:
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, estimator in regression_models().items():
        mae, rmse, r2 = [], [], []
        for train_idx, test_idx in cv.split(X):
            model = clone(estimator)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            mae.append(mean_absolute_error(y[test_idx], pred))
            rmse.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
            r2.append(r2_score(y[test_idx], pred))
        results[name] = {"MAE": summarize(mae), "RMSE": summarize(rmse), "R2": summarize(r2)}
    return results


def evaluate_classification(X: pd.Series, score_labels: np.ndarray) -> dict:
    y = (score_labels >= 4).astype(int)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, estimator in classification_models().items():
        accuracy, f1, auc = [], [], []
        for train_idx, test_idx in cv.split(X, y):
            model = clone(estimator)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            decision = model.decision_function(X.iloc[test_idx])
            accuracy.append(accuracy_score(y[test_idx], pred))
            f1.append(f1_score(y[test_idx], pred, zero_division=0))
            auc.append(roc_auc_score(y[test_idx], decision))
        results[name] = {"Accuracy": summarize(accuracy), "F1": summarize(f1), "ROC_AUC": summarize(auc)}
    return results


def main() -> None:
    df = load_dataset()
    X, y = df["text"], df["label"].to_numpy(dtype=float)
    regression = evaluate_regression(X, y)
    classification = evaluate_classification(X, y)
    best_regressor = min(regression, key=lambda n: regression[n]["MAE"]["mean"])
    best_classifier = max(classification, key=lambda n: classification[n]["F1"]["mean"])
    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"rows": int(len(df)), "label_min": float(y.min()), "label_max": float(y.max())},
        "cv": {"folds": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE},
        "regression": regression, "classification": classification,
        "recommended_regressor": best_regressor, "recommended_classifier": best_classifier,
    }
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    path = EXPERIMENT_DIR / "latest.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved experiment report to {path}")


if __name__ == "__main__":
    main()
