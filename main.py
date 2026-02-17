from src.features import create_tfidf_features
import pandas as pd
from src.preprocess import clean_text
from src.model import train_model
from src.model import predict_question
from src.model import rank_questions

# Load dataset
data = pd.read_csv("data/questions.csv")

# Apply cleaning
data["cleaned_question"] = data["question"].apply(clean_text)

print("Cleaned Data ✅")
print(data[["question", "cleaned_question"]])

# Convert text to TF-IDF features
X, vectorizer = create_tfidf_features(data["cleaned_question"])

print("\nTF-IDF Shape:", X.shape)

# Labels (target values)
y = data["label"]

# Train the model
model = train_model(X, y)

print("\nModel trained successfully ✅")

# ---- Test LOQQIN on a new question ----
new_question = "Explain gradient descent algorithm"

prediction, confidence = predict_question(
    model,
    vectorizer,
    [new_question][0]   # keeps format simple
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
