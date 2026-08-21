"""Train the final LOQQIN regression model on all labeled data.

Use ``python evaluate.py`` first to benchmark candidate models with 5-fold CV.
This script then fits the selected StackingRegressor on the complete dataset and
saves ``model.pkl`` and ``vectorizer.pkl`` for the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import create_tfidf_features
from src.model import train_model
from src.preprocess import clean_text
from src.utils import save_objects

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "questions.csv"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = {"question", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["question", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise")
    if not df["label"].between(0, 10).all():
        raise ValueError("All labels must be between 0 and 10")
    return df


def main() -> None:
    df = load_dataset()
    df["cleaned_question"] = df["question"].map(clean_text)
    df = df[df["cleaned_question"].str.strip().ne("")].copy()

    X, vectorizer = create_tfidf_features(df["cleaned_question"])
    y = df["label"].to_numpy(dtype=float)
    model = train_model(X, y)
    save_objects(model, vectorizer)

    print(f"Trained on {len(df)} questions")
    print(f"Label range: {y.min():.0f}-{y.max():.0f}")
    print("Saved model.pkl and vectorizer.pkl")


if __name__ == "__main__":
    main()
