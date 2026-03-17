import pandas as pd
import joblib
import sys
import os

# Ensure src folder is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import clean_text
from src.features import create_tfidf_features
from src.model import train_model

def main():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Load Data
    try:
        # ✅ Best Practice (use os.path)
       csv_path = os.path.join(os.path.dirname(__file__), 'data', 'questions.csv')
       df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
       print(f"✅ Loaded {len(df)} questions.")
    except FileNotFoundError:
        print("❌ Error: questions.csv not found!")
        return

    # 2. Preprocess Text (Must match app.py logic)
    print("🧹 Cleaning text...")
    df['cleaned_question'] = df['question'].apply(clean_text)
    
    # 3. Create Features
    print("🔢 Vectorizing features...")
    X, vectorizer = create_tfidf_features(df['cleaned_question'].values)
    y = df['label'].values
    
    # Check labels
    unique_labels = set(y)
    print(f"🏷️  Unique labels found: {unique_labels}")
    
    if unique_labels == {0, 1}:
        print("ℹ️  Detected Binary Labels (0/1). Score will be Probability of High Quality.")
    else:
        print("⚠️  Detected Multi-class Labels. Ensure model.py handles indexing correctly.")

    # 4. Train Model
    print("🧠 Training model (this may take a minute)...")
    model = train_model(X, y)
    
    # 5. Save Artifacts
    print("💾 Saving model.pkl and vectorizer.pkl...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("✅ Training Complete! You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    main()