import os
import pandas as pd

from src.features import create_tfidf_features
from src.preprocess import clean_text
from src.model import train_model, predict_question, rank_questions
from src.utils import save_objects, load_objects


# Load dataset
data = pd.read_csv("data/questions.csv")

# Apply cleaning
data["cleaned_question"] = data["question"].apply(clean_text)

print("Cleaned Data ✅")
print(data[["question", "cleaned_question"]])

# Convert text to TF-IDF features
X, vectorizer = create_tfidf_features(data["cleaned_question"])

print("\nTF-IDF Shape:", X.shape)


# -------------------------------
# AUTO LOAD OR TRAIN MODEL
# -------------------------------
if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    print("\nLoading existing LOQQIN brain 🧠")
    model, vectorizer = load_objects()

else:
    print("\nTraining LOQQIN for first time 🚀")

    y = data["label"]
    model = train_model(X, y)

    save_objects(model, vectorizer)
    print("Model trained and saved ✅")


# ---- Test LOQQIN on a new question ----
new_question = "Explain gradient descent algorithm"

prediction, confidence = predict_question(
    model,
    vectorizer,
    new_question
)

print("\nNew Question:", new_question)

if prediction == 1:
    print("Prediction: Deep Conceptual Question ✅")
else:
    print("Prediction: Surface Level Question")

print("Confidence:", round(confidence, 2))


# ---- Rank multiple questions ----
test_questions = [
    "Define artificial intelligence",
    "Explain gradient descent algorithm",
    "List types of machine learning",
    "Compare supervised and unsupervised learning"
]

ranked_output = rank_questions(model, vectorizer, test_questions)

print("\nLOQQIN Ranked Questions 🔥")
for q, score in ranked_output:
    print(f"{q}  --> Score: {round(score,2)}")
