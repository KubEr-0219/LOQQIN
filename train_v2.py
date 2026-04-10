# -----------------------------
# TRAINING SCRIPT V2 - Better Label Thresholds
# -----------------------------
import pandas as pd
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.features import create_tfidf_features
from src.model import train_model
from src.utils import save_objects
from sklearn.model_selection import cross_val_score  # ✅ Fixed: only cross_val_score here
from sklearn.metrics import classification_report, confusion_matrix  # ✅ Fixed: classification_report here

# -----------------------------
# LOAD DATA
# -----------------------------
print("📊 Loading training data...")
df = pd.read_csv("questions.csv")

# -----------------------------
# BETTER LABEL THRESHOLD
# -----------------------------
# ✅ 0-3 = Low Quality (binary label 0)
# ✅ 4-10 = High Quality (binary label 1)
# This creates better class separation
df['binary_label'] = (df['label'] >= 4).astype(int)

print(f"\n📈 Class Distribution:")
print(df['binary_label'].value_counts().to_dict())
print(f"Ratio: {df['binary_label'].sum()} high / {len(df) - df['binary_label'].sum()} low")

# -----------------------------
# PREPARE FEATURES
# -----------------------------
print("\n🔧 Creating TF-IDF features...")
X, vectorizer = create_tfidf_features(df['question'].tolist())
y = df['binary_label'].tolist()

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\n🤖 Training model...")
model = train_model(X, y)

# -----------------------------
# EVALUATE
# -----------------------------
print("\n📊 Evaluation:")
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Predict on training data for inspection
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

print(f"\n📈 Probability Distribution:")
print(f"Mean: {y_proba.mean():.3f}")
print(f"Min: {y_proba.min():.3f}")
print(f"Max: {y_proba.max():.3f}")
print(f"Std: {y_proba.std():.3f}")

print(f"\n📋 Probability Ranges:")
print(f"0.0-0.3: {sum(p < 0.3 for p in y_proba)} questions")
print(f"0.3-0.6: {sum(0.3 <= p < 0.6 for p in y_proba)} questions")
print(f"0.6-0.9: {sum(0.6 <= p < 0.9 for p in y_proba)} questions")
print(f"0.9-1.0: {sum(p >= 0.9 for p in y_proba)} questions")

print("\n" + "="*50)
print(classification_report(y, y_pred, target_names=['Low Quality', 'High Quality']))
print("="*50)

# -----------------------------
# SAVE MODEL
# -----------------------------
print("\n💾 Saving model and vectorizer...")
save_objects(model, vectorizer)
print("✅ Done! Model files saved: model.pkl, vectorizer.pkl")

# -----------------------------
# TEST SAMPLE QUESTIONS
# -----------------------------
print("\n" + "="*50)
print("🧪 Testing Sample Questions:")
print("="*50)

from src.preprocess import clean_text
from src.model import predict_question

test_questions = [
    ("What is IoT?", 2, 4),
    ("Define cloud computing", 2, 4),
    ("Explain IoT architecture", 6, 8),
    ("Compare M2M and IoT", 7, 9),
    ("Design a secure IoT system with edge computing", 8, 10),
]

for q, expected_min, expected_max in test_questions:
    cleaned = clean_text(q)
    pred, score = predict_question(model, vectorizer, cleaned)
    status = "✅" if expected_min <= score <= expected_max else "⚠️"
    print(f"{status} {q[:50]}... → {score}/10 (expected {expected_min}-{expected_max})")