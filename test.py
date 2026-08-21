#!/usr/bin/env python3
"""Small smoke test for the trained LOQQIN artifacts."""

from src.model import analyze_question_metrics, predict_question
from src.preprocess import clean_text
from src.utils import load_objects


def main() -> None:
    model, vectorizer = load_objects()
    samples = [
        "What is IoT?",
        "Explain IoT architecture",
        "Compare edge and cloud processing",
        "Design a secure IoT system with edge computing",
    ]

    for question in samples:
        cleaned = clean_text(question)
        prediction, score = predict_question(model, vectorizer, cleaned)
        clarity, specificity, bloom = analyze_question_metrics(question)
        assert 0.0 <= score <= 10.0
        print(f"{score:>5.2f}/10 | class={prediction} | Bloom={bloom:<9} | {question}")
        print(f"           clarity={clarity}, specificity={specificity}")


if __name__ == "__main__":
    main()
