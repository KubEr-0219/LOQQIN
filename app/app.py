# -----------------------------
# IMPORTS
# -----------------------------
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import joblib
from src.preprocess import clean_text
from src.model import predict_question, rank_questions, analyze_question_metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please train the model first and ensure model.pkl and vectorizer.pkl are in the same folder as app.py")
    st.stop()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="LOQQIN",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# DESIGN SYSTEM (CSS)
# -----------------------------
st.markdown("""
<style>

body {
    background-color: #F9F9F7;
}

.main {
    max-width: 780px;
    margin: auto;
}

h1, h2, h3 {
    font-family: "Inter", sans-serif;
    color: #1A1A2E;
}

p, label {
    font-family: "Inter", sans-serif;
    color: #6B7280;
}

/* Card */
.loqqin-card {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.05);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* Button */
.stButton>button {
    background-color: #4F46E5;
    color: white;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    border: none;
}

.stButton>button:hover {
    background-color: #4338CA;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("# 🧠 LOQQIN")
st.caption("Learning‑Oriented Question Quality Predictor")
st.divider()
st.write("Analyze and rank exam questions using Machine Learning.")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["Single Question Analyzer", "Batch Upload"])

# =====================================================
# TAB 1 — SINGLE QUESTION (FIXED + POLISHED)
# =====================================================
with tab1:

    # ✅ Proper card wrapper (no broken card function)

    st.subheader("Analyze a Question")

    question = st.text_area(
        "Enter your question",
        placeholder="Explain the architecture of IoT..."
    )

    if st.button("Analyze Question", use_container_width=True):

        if not question.strip():
            st.warning("⚠️ Please enter a question before analyzing.")
            st.stop()

        with st.spinner("Analyzing question depth…"):

            cleaned = clean_text(question)
            prediction, score = predict_question(model, vectorizer, cleaned)

        score10 = round(score, 1)

        st.markdown("### Quality Score")

        # ✅ progress (correct scale)
        st.progress(float(max(0.0, min(score, 10.0))) / 10.0)

        # ✅ big centered score
        st.markdown(
            f"<h2 style='text-align:center'>{score10}/10</h2>",
            unsafe_allow_html=True
        )

        # ✅ quality label
        if score10 >= 7:
            st.success("High Quality Question ⭐")
        elif score10 >= 4:
            st.warning("Medium Depth Question")
        else:
            st.error("Surface Level Question")

        # ✅ insight metrics
        col1, col2, col3 = st.columns(3)

        clarity, specificity, bloom_level = analyze_question_metrics(question)

        col1.metric("Clarity", clarity)
        col2.metric("Specificity", specificity)
        col3.metric("Bloom Level", bloom_level)

    
# =====================================================
# TAB 2 — BATCH UPLOAD (CLEANED)
# =====================================================
with tab2:

    uploaded_file = st.file_uploader(
        "Upload a .txt file containing questions",
        type=["txt"]
    )

    if uploaded_file:

        content = uploaded_file.read().decode("utf-8")

        questions = [q.strip() for q in content.split("\n") if q.strip()]

        st.write(f"Questions detected: {len(questions)}")

        cleaned_questions = [clean_text(q) for q in questions]

# Zip originals with cleaned so we can display originals
        question_pairs = list(zip(questions, cleaned_questions))

        ranked = rank_questions(model, vectorizer, [c for _, c in question_pairs])

# Map cleaned back to originals for display
        cleaned_to_original = dict(zip(cleaned_questions, questions))

        st.subheader("Ranked Questions")

        for q, prediction, score in ranked:
            original = cleaned_to_original.get(q, q)  # fallback to q if not found

            score10 = round(score, 1)

            st.markdown(f"### {original}")

            st.progress(float(max(0.0, min(score, 10.0))) / 10.0)

            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown(f"**{score10}/10**")

            with col2:
                if score10 >= 7:
                    st.success("High Quality ⭐")
                elif score10 >= 4:
                    st.info("Medium Quality")
                else:
                    st.warning("Low Quality")

            st.divider()