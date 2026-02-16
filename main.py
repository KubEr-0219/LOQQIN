from src.features import create_tfidf_features
import pandas as pd
from src.preprocess import clean_text

# Load dataset
data = pd.read_csv("data/questions.csv")

# Apply cleaning
data["cleaned_question"] = data["question"].apply(clean_text)

print("Cleaned Data ✅")
print(data[["question", "cleaned_question"]])

# Convert text to TF-IDF features
X, vectorizer = create_tfidf_features(data["cleaned_question"])

print("\nTF-IDF Shape:", X.shape)