"""Model training, inference, and question-quality utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


def build_regressor(random_state: int = 42) -> StackingRegressor:
    """Build the production regression ensemble used by LOQQIN."""
    estimators = [
        ("ridge", Ridge(alpha=1.0)),
        ("gbr", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state)),
        ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ]
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=0.5), cv=3, n_jobs=-1)


def train_model(X, y, random_state: int = 42) -> StackingRegressor:
    """Fit the production regression ensemble."""
    model = build_regressor(random_state=random_state)
    model.fit(X, y)
    return model


def _clamp_score(score: float) -> float:
    return float(np.clip(score, 0.0, 10.0))


def predict_question(model, vectorizer, question: str) -> tuple[int, float]:
    """Predict the learned 0-10 quality score without keyword overrides."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    question_vector = vectorizer.transform([question.strip()])
    ml_score = _clamp_score(float(model.predict(question_vector)[0]))
    prediction = int(ml_score >= 4.0)
    return prediction, round(ml_score, 2)


def heuristic_score(question: str) -> float:
    """Return an optional explanatory heuristic; never alter the ML prediction."""
    q = question.lower().strip()
    weights = {
        "design": 3.0, "architect": 3.0, "develop": 2.5, "create": 3.0, "formulate": 2.0,
        "evaluate": 2.5, "assess": 2.0, "critique": 2.5, "justify": 2.0,
        "analyze": 2.5, "compare": 2.0, "contrast": 2.0, "differentiate": 2.0,
        "explain": 1.5, "discuss": 1.5, "describe": 1.0, "summarize": 0.5,
        "apply": 1.5, "implement": 1.5, "solve": 1.0,
        "define": -2.0, "what is": -1.5, "what are": -1.5, "list": -2.0,
        "name": -1.5, "state": -1.0, "identify": -1.0, "recall": -1.5,
        "architecture": 1.0, "algorithm": 0.8, "trade-off": 1.0, "tradeoff": 1.0,
        "complexity": 0.5, "optimization": 0.5,
    }
    score = sum(weight for term, weight in weights.items() if term in q)
    words = len(q.split())
    score += 0.5 if words >= 12 else -1.0 if words < 4 else 0.0
    return float(np.clip(score, -5.0, 5.0))


def rank_questions(model, vectorizer, questions: Sequence[str]) -> list[dict]:
    """Score and rank questions while preserving original text."""
    results = []
    for question in questions:
        _, score = predict_question(model, vectorizer, question)
        results.append({
            "question": question,
            "prediction": int(score >= 4.0),
            "score": score,
            "rule_signal": round(heuristic_score(question), 2),
            "length": len(question.strip().split()),
        })
    return sorted(results, key=lambda item: item["score"], reverse=True)


def analyze_question_metrics(question: str) -> tuple[str, str, str]:
    """Compute transparent deterministic clarity, specificity, and Bloom level."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    q = question.lower().strip()
    tokens = set(q.replace("?", " ").split())
    bloom_levels = {
        "Create": {"design", "construct", "develop", "formulate", "author", "create", "build", "architect"},
        "Evaluate": {"evaluate", "critique", "justify", "defend", "judge", "recommend", "assess"},
        "Analyze": {"analyze", "analyse", "compare", "contrast", "differentiate", "examine"},
        "Apply": {"apply", "solve", "use", "demonstrate", "calculate", "implement", "execute"},
        "Understand": {"explain", "describe", "summarize", "interpret", "classify", "discuss"},
        "Remember": {"define", "list", "name", "state", "recall", "identify", "what", "who", "when"},
    }
    bloom_level = "Remember"
    for level in ("Create", "Evaluate", "Analyze", "Apply", "Understand"):
        if tokens.intersection(bloom_levels[level]):
            bloom_level = level
            break
    word_count = len(q.split())
    clarity = "Too Short" if word_count < 5 else "High" if word_count <= 12 else "Medium" if word_count <= 25 else "Verbose"
    technical_terms = {"architecture", "algorithm", "protocol", "mechanism", "system", "framework", "network", "database", "optimization"}
    overlap = tokens.intersection(technical_terms)
    specificity = "High" if len(overlap) >= 2 else "Medium" if overlap else "Low"
    return clarity, specificity, bloom_level
