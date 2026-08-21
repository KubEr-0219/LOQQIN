"""Artifact persistence helpers."""

from __future__ import annotations

from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "vectorizer.pkl"


def save_objects(model, vectorizer) -> None:
    """Persist the trained model and vectorizer at deterministic project paths."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)


def load_objects():
    """Load the trained model and vectorizer."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("model.pkl/vectorizer.pkl not found. Run `python train.py` first.")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
