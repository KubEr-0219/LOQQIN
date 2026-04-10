#!/usr/bin/env python3
import joblib
from src.preprocess import clean_text
from src.model import predict_question

print("🔍 Loading model...")
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("❌ Run 'python train.py' first!")
    exit(1)

print("\n" + "="*70)
print("LOQQIN - Score Verification")
print("="*70)

tests = [
    ("What is IoT?", 2, 4),
    ("Define cloud computing", 2, 4),
    ("Describe how MQTT works", 4, 6),
    ("Explain IoT architecture", 6, 8),
    ("Compare M2M and IoT", 7, 9),
    ("Design a secure IoT system with edge computing", 8, 10),
]

print(f"\n{'Score':<8} {'Expected':<12} {'Question'}")
print("-"*70)

passed = 0
for question, exp_min, exp_max in tests:
    cleaned = clean_text(question)
    pred, score = predict_question(model, vectorizer, cleaned)
    in_range = exp_min <= score <= exp_max
    if in_range: passed += 1
    status = "✅" if in_range else "⚠️"
    print(f"{status} {score:4.1f}/10   {exp_min}-{exp_max:4}   {question}")

print("-"*70)
print(f"Results: {passed}/{len(tests)} in expected range")