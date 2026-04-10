#!/usr/bin/env python3
"""
LOQQIN Training Script - REGRESSION VERSION
Treats labels 0-10 as continuous quality scores instead of binary.
"""
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text
from src.features import create_tfidf_features
from src.utils import save_objects

print("📊 Loading questions.csv...")
df = pd.read_csv("questions.csv")

# ✅ REGRESSION: Use raw labels 0-10 directly (no binarization)
# This fixes the issue where "Explain..." (label 1) was treated as negative
y = df['label'].values
print(f"Score range: {y.min()} - {y.max()}")
print(f"Mean score: {y.mean():.2f}")

# ✅ Apply clean_text()
print("🔧 Preprocessing with clean_text()...")
df['cleaned'] = df['question'].apply(clean_text)
df = df[df['cleaned'].str.len() > 0]

# ✅ Create TF-IDF features
print("🚀 Creating TF-IDF features...")
X, vectorizer = create_tfidf_features(df['cleaned'].tolist())

print(f"   Feature matrix: {X.shape}")

# ✅ Train Regression Ensemble (better for 0-10 scale)
print("🤖 Training regression ensemble...")

# Base learners
ridge = Ridge(alpha=1.0, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# Stacking ensemble
estimators = [
    ('ridge', ridge),
    ('gbr', gbr),
    ('svr', svr)
]

model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=0.5),
    cv=3
)

model.fit(X, y)

# Save
print("💾 Saving model.pkl and vectorizer.pkl...")
save_objects(model, vectorizer)

# Quick verification test
print("\n🧪 Quick score verification:")
from src.model import predict_question

tests = [
    ("What is IoT?", 0, 3),
    ("Explain IoT architecture", 5, 8),
    ("Design a secure IoT system with edge computing", 7, 10),
    ("Analyze trade-offs between edge and cloud processing", 6, 9),
]

for q, exp_min, exp_max in tests:
    cleaned = clean_text(q)
    _, score = predict_question(model, vectorizer, cleaned)
    status = "✅" if exp_min <= score <= exp_max else "⚠️"
    print(f"{status} {score:4.1f}/10 | {q[:50]}")

print("\n✨ Done! Run 'streamlit run app.py' to use LOQQIN.")