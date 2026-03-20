import joblib
import os

print("🔍 LOQQIN Diagnostic Test\n")

# 1. Check file locations
print("1. Checking file locations...")
files_to_check = [
    "model.pkl",
    "vectorizer.pkl",
    "app.py",
    "src/model.py",
    "src/preprocess.py",
    "src/features.py"
]

for f in files_to_check:
    exists = os.path.exists(f)
    status = "✅" if exists else "❌"
    print(f"   {status} {f}")

# 2. Try loading models
print("\n2. Loading model files...")
try:
    model = joblib.load("model.pkl")
    print(f"   ✅ model.pkl loaded - Type: {type(model).__name__}")
except Exception as e:
    print(f"   ❌ model.pkl failed: {e}")

try:
    vectorizer = joblib.load("vectorizer.pkl")
    print(f"   ✅ vectorizer.pkl loaded - Type: {type(vectorizer).__name__}")
except Exception as e:
    print(f"   ❌ vectorizer.pkl failed: {e}")

# 3. Test prediction
print("\n3. Testing prediction...")
try:
    from src.preprocess import clean_text
    from src.model import predict_question
    
    test_q = "what is iot"
    cleaned = clean_text(test_q)
    pred, score = predict_question(model, vectorizer, cleaned)
    print(f"   ✅ Test question: '{test_q}'")
    print(f"   ✅ Cleaned: '{cleaned}'")
    print(f"   ✅ Score: {score}/10")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Diagnostic Complete!")