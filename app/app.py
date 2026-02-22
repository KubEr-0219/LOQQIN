import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# -----------------------------
# IMPORTS
# -----------------------------
import streamlit as st
import joblib

from src.preprocess import clean_text
from src.model import predict_question, rank_questions

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    font-weight: 600;
}

.stButton>button {
    border-radius: 10px;
    padding: 0.5em 1.2em;
    font-weight: 500;
}

.stTextInput>div>div>input {
    border-radius: 10px;
}

.stTextArea textarea {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# LOAD LOQQIN BRAIN
# -----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# -----------------------------
# APP TITLE
# -----------------------------
st.title("🧠 LOQQIN")
st.caption("Smart Question Analysis Tool for Students")
st.write("Analyze and rank exam questions using Machine Learning.")


# =====================================================
# SINGLE QUESTION PREDICTION
# =====================================================
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

# =====================================================
# FILE UPLOAD RANKING
# =====================================================
st.header("Upload Question Paper")

uploaded_file = st.file_uploader(
    "Upload a .txt file containing questions",
    type=["txt"]
)

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")

    questions = [q.strip() for q in content.split("\n") if q.strip()]
    st.write("Questions detected:", len(questions))

    cleaned_questions = [clean_text(q) for q in questions]
    ranked = rank_questions(model, vectorizer, cleaned_questions)

    st.subheader("Ranked Questions 🔥")

    for q, score in ranked:
        score10 = round(score * 10, 1)
        st.write(f"{q} — Score: {score10}/10")

# ---- Top Recommended Questions ----
# show only after ranking exists
if "ranked" in locals():

    st.subheader("🏆 Top Recommended Questions")

    top_questions = ranked[:3]   # top 3 highest scores

    for q, score in top_questions:
        score10 = round(score * 10, 1)
        st.success(f"⭐ {q}  (Score: {score10}/10)")
    