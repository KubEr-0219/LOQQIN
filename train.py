import pandas as pd
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import clean_text
from src.features import create_tfidf_features
from src.model import train_model

def main():
    print("🚀 Starting Training Pipeline...")
    
    try:
        df = pd.read_csv('questions.csv')
        print(f"✅ Loaded {len(df)} questions.")
    except FileNotFoundError:
        print("❌ Error: questions.csv not found!")
        return

    print("🧹 Cleaning text...")
    df['cleaned_question'] = df['question'].apply(clean_text)
    
    print("🔢 Vectorizing features...")
    X, vectorizer = create_tfidf_features(df['cleaned_question'].values)
    y = df['label'].values
    
    unique_labels = set(y)
    print(f"🏷️  Unique labels found: {unique_labels}")
    
    print("🧠 Training model (this may take a minute)...")
    model = train_model(X, y)
    
    print("💾 Saving model.pkl and vectorizer.pkl...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("✅ Training Complete! You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    main()