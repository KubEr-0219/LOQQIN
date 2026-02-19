import streamlit as st
import joblib
import sys
import os

# Adding project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import clean_text
from src.model import predict_question, rank_questions

# Load saved LOQQIN brain
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🧠 LOQQIN — Question Quality Analyzer")

st.write("Analyze and rank exam questions using Machine Learning.")

# -------------------------
# Single Question Prediction
# -------------------------
st.header("Predict Question Quality")

user_question = st.text_input("Enter a question:")

if st.button("Analyze Question"):
    cleaned = clean_text(user_question)
    prediction, confidence = predict_question(
        model, vectorizer, cleaned
    )

    if prediction == 1:
        st.success("Deep Conceptual Question ✅")
    else:
        st.warning("Surface Level Question")

    st.write("Confidence:", round(confidence, 2))


# -------------------------
# Multiple Question Ranking
# -------------------------
st.header("Rank Multiple Questions")

multi_questions = st.text_area(
    "Enter multiple questions (one per line):"
)

if st.button("Rank Questions"):
    questions = [q for q in multi_questions.split("\n") if q.strip()]

    cleaned_questions = [clean_text(q) for q in questions]

    ranked = rank_questions(model, vectorizer, cleaned_questions)

    st.subheader("Ranked Questions 🔥")

    for q, score in ranked:
        st.write(f"{q} — Score: {round(score,2)}")
