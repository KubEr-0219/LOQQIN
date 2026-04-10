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

# Load Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run `python train.py` first.")
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
# MERCHBANAO DESIGN SYSTEM (CSS)
# ✅ UPDATED: Button aesthetic with solid bottom shadow effect
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --bg: #F7F5F0;
        --bg-card: #FFFFFF;
        --bg-elevated: #EFECE6;
        --text-primary: #1A1A1A;
        --text-secondary: #4A4A4A;
        --text-muted: #9A9A9A;
        --accent: #1A1A1A;
        --accent-glow: rgba(26, 26, 26, 0.06);
        --border: #E2DED6;
        --border-active: #B8B2A8;
        --success: #16A34A;
        --warning: #D97706;
        --destructive: #DC2626;
        --radius: 14px;
        --radius-sm: 10px;
    }

    html, body, .stApp {
        background-color: var(--bg) !important;
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* ── HEADER ── */
    .loqqin-header {
        text-align: center;
        padding: 56px 0 40px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 36px;
        animation: fadeInUp 0.5s ease-out both;
    }
    .loqqin-header h1 {
        font-family: 'Syne', sans-serif;
        font-size: 52px;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -1.5px;
        margin: 0 0 12px 0;
    }
    .loqqin-header p {
        color: var(--text-muted);
        font-size: 14px;
        font-weight: 400;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin: 0;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-card);
        padding: 6px;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        margin-bottom: 32px;
        animation: fadeInUp 0.5s ease-out 0.15s both;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 14px;
        border-radius: var(--radius-sm);
        background: transparent;
        border: none;
        color: var(--text-muted);
        padding: 10px 22px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary);
        background: var(--bg-elevated);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--accent);
        color: #FFFFFF;
        font-weight: 700;
    }

    /* ── SECTION TITLES ── */
    h2, h3, .stSubheader {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.5px;
    }
    .stCaption, p.caption {
        color: var(--text-muted) !important;
        font-size: 13px !important;
    }

    /* ── TEXTAREA ── */
    .stTextArea label {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 13px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTextArea textarea {
        font-family: 'DM Sans', sans-serif;
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-size: 15px;
        padding: 16px;
        transition: border-color 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: var(--border-active) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* ── BUTTONS ── ✅ UPDATED WITH SHADOW AESTHETIC */
    .stButton > button,
    .stDownloadButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 14px 28px !important;
        width: 100% !important;
        height: 50px !important;
        letter-spacing: 0.2px;
        position: relative !important;
        margin-bottom: 4px !important;
        box-shadow: 0 4px 0 #2d2d2d !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 0 #2d2d2d !important;
    }
    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(4px) !important;
        box-shadow: 0 0 0 #2d2d2d !important;
    }
    .stButton > button[kind="secondary"],
    .stDownloadButton > button[kind="secondary"] {
        background: var(--bg-elevated) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 4px 0 #2d2d2d !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stDownloadButton > button[kind="secondary"]:hover {
        border-color: var(--border-active) !important;
        color: var(--text-primary) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 0 #2d2d2d !important;
    }
    .stButton > button[kind="secondary"]:active,
    .stDownloadButton > button[kind="secondary"]:active {
        transform: translateY(4px) !important;
        box-shadow: 0 0 0 #2d2d2d !important;
    }

    /* ── PROGRESS BAR ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--destructive), var(--warning), var(--success));
        border-radius: 999px;
    }
    .stProgress > div {
        height: 6px;
        border-radius: 999px;
        background: var(--bg-elevated);
        border: none;
        padding: 0;
    }
    .stProgress {
        margin: 12px 0 !important;
    }
    .stProgress > div {
        height: 5px !important;
        padding: 0 !important;
        background: var(--bg-elevated) !important;
        border: none !important;
    }

    /* ── SCORE DISPLAY ── */
    .score-display {
        font-family: 'Syne', sans-serif;
        font-size: 80px;
        font-weight: 800;
        color: var(--text-primary);
        text-align: center;
        letter-spacing: -3px;
        line-height: 1;
        margin: 24px 0 8px 0;
        animation: scoreReveal 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .score-display span {
        font-size: 28px;
        font-weight: 400;
        color: var(--text-muted);
        letter-spacing: 0;
    }

    /* ── BADGE ── */
    .badge {
        display: block;
        padding: 12px 24px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.3px;
        border: 1px solid;
        width: 100%;
        animation: fadeInUp 0.4s ease-out 0.1s both;
    }
    .badge-high { background: rgba(22, 163, 74, 0.08); color: var(--success); border-color: rgba(22, 163, 74, 0.3); }
    .badge-med  { background: rgba(217, 119, 6, 0.08); color: var(--warning); border-color: rgba(217, 119, 6, 0.3); }
    .badge-low  { background: rgba(220, 38, 38, 0.08); color: var(--destructive); border-color: rgba(220, 38, 38, 0.3); }

    /* ── METRIC CARDS ── */
    .metric-card {
        background: var(--bg-card);
        padding: 24px 16px;
        border-radius: var(--radius-sm);
        text-align: center;
        border: 1px solid var(--border);
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeInUp 0.4s ease-out both;
    }
    .metric-card:hover {
        transform: translateY(-4px) !important;
        border-color: var(--border-active) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
    }
    .metric-label {
        color: var(--text-muted);
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .metric-value {
        color: var(--text-primary);
        font-family: 'Syne', sans-serif;
        font-size: 20px;
        font-weight: 700;
    }

    /* ── STAT CARDS ── */
    .stat-card {
        background: var(--bg-card);
        padding: 20px 12px;
        border-radius: var(--radius-sm);
        text-align: center;
        border: 1px solid var(--border);
        animation: fadeInUp 0.4s ease-out both;
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08) !important;
    }
    .stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 32px;
        font-weight: 800;
        color: var(--text-primary);
    }
    .stat-label {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 6px;
    }

    /* ── CHAR COUNTER ── */
    .char-counter {
        color: var(--text-muted);
        font-size: 12px;
        font-weight: 500;
        margin-top: 6px;
    }

    /* ── EXPANDER ── */
    [data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }
    [data-testid="stExpander"] > details > summary {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stExpander"] > details[open] > div {
        background: var(--bg-card) !important;
        border-top: 1px solid var(--border) !important;
    }
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        animation: fadeIn 0.3s ease-out both;
        transition: background 0.2s ease, border-color 0.2s ease !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: var(--border-active) !important;
        background: var(--bg-elevated) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
    }

    /* ── ALERTS ── */
    .stAlert {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-secondary) !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] > div {
        background: var(--bg-card) !important;
        border: 1px dashed var(--border-active) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stFileUploader"] * {
        color: var(--text-secondary) !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background: var(--bg-card) !important;
    }
    button[data-testid="baseButton-secondary"] {
        background: var(--bg-elevated) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-active) !important;
    }

    /* ── DIVIDER ── */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 28px 0;
    }

    /* ── SPINNER ── */
    .stSpinner > div { border-top-color: var(--accent) !important; }

    /* ── FOOTER ── */
    .loqqin-footer {
        text-align: center;
        margin-top: 60px;
        color: var(--text-muted);
        font-size: 12px;
        padding: 24px;
        border-top: 1px solid var(--border);
        letter-spacing: 0.3px;
    }

    /* ── KEYFRAMES ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes scoreReveal {
        from { opacity: 0; transform: translateY(24px) scale(0.95); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    .stTabs [data-baseweb="tab-panel"] {
        animation: fadeInUp 0.5s ease-out 0.25s both;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="loqqin-header">
    <h1>LOQQIN</h1>
    <p>Learning-Oriented Question Quality Predictor</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["Single Analyzer", "Batch Upload"])

# =====================================================
# TAB 1: SINGLE QUESTION
# =====================================================
with tab1:
    st.subheader("Analyze a Question")
    st.caption("Paste any exam or assessment question to get instant quality feedback.")
    
    question = st.text_area(
        "Enter your question",
        placeholder="Explain the architecture of IoT and how sensors communicate with the cloud...",
        height=140
    )
    
    st.markdown(f'<p class="char-counter">{len(question)} characters</p>', unsafe_allow_html=True)
    
    if st.button("✨ Analyze Question"):
        if not question.strip():
            st.warning("⚠️ Please enter a question to analyze.")
        else:
            with st.spinner("🔮 Analyzing question depth..."):
                cleaned = clean_text(question)
                prediction, score = predict_question(model, vectorizer, cleaned)
                clarity, specificity, bloom = analyze_question_metrics(question)
            
            score10 = round(score, 1)
            
            # Score Section
            st.markdown('<div style="text-align: center; margin: 40px 0; animation: fadeInUp 0.5s ease-out both;">', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">✨ Quality Score</p>', unsafe_allow_html=True)
            st.progress(min(score, 10.0) / 10.0)
            st.markdown(f'<div class="score-display">{score10}<span>/10</span></div>', unsafe_allow_html=True)
            
            # Badge
            if score10 >= 7:
                badge_class = "badge-high"
                badge_text = "High Quality Question"
            elif score10 >= 4:
                badge_class = "badge-med"
                badge_text = "Medium Depth Question"
            else:
                badge_class = "badge-low"
                badge_text = "Surface Level Question"
            
            st.markdown(f'<div class="badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Metrics Grid
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'''
                <div class="metric-card" style="animation-delay: 0.05s;">
                    <div class="metric-label">👁️ Clarity</div>
                    <div class="metric-value">{clarity}</div>
                </div>
                ''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''
                <div class="metric-card" style="animation-delay: 0.15s;">
                    <div class="metric-label">🎯 Specificity</div>
                    <div class="metric-value">{specificity}</div>
                </div>
                ''', unsafe_allow_html=True)
            with c3:
                st.markdown(f'''
                <div class="metric-card" style="animation-delay: 0.25s;">
                    <div class="metric-label">📚 Bloom Level</div>
                    <div class="metric-value">{bloom}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Analyze Another", use_container_width=True):
                    st.rerun()
            with c2:
                # ✅ Export button now matches (black background, white text)
                st.download_button(
                    label="📥 Export Result",
                    data=f"LOQQIN Analysis Report\n\nQuestion: {question}\nScore: {score10}/10\nClarity: {clarity}\nSpecificity: {specificity}\nBloom Level: {bloom}\nVerdict: {badge_text}",
                    file_name="loqqin-report.txt",
                    use_container_width=True
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# TAB 2: BATCH UPLOAD
# =====================================================
with tab2:
    st.subheader("📦 Batch Question Analysis")
    st.caption("Upload multiple questions to rank by quality. One question per line.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"], label_visibility="collapsed")
    
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        questions = [q.strip() for q in content.split("\n") if q.strip()]
        
        st.markdown(f'**📊 Questions detected:** {len(questions)}')
        
        if st.button(f"✨ Analyze {len(questions)} Questions", use_container_width=True):
            with st.spinner("🔮 Analyzing questions..."):
                cleaned_questions = [clean_text(q) for q in questions]
                ranked = rank_questions(model, vectorizer, cleaned_questions)
            
            st.success(f"✅ Analyzed {len(questions)} questions!")
            
            # Stats
            high = sum(1 for r in ranked if r['score'] >= 7)
            medium = sum(1 for r in ranked if 4 <= r['score'] < 7)
            low = sum(1 for r in ranked if r['score'] < 4)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'''
                <div class="stat-card" style="animation-delay: 0.05s;">
                    <div class="stat-value">{len(questions)}</div>
                    <div class="stat-label">📋 Total</div>
                </div>
                ''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''
                <div class="stat-card" style="animation-delay: 0.12s;">
                    <div class="stat-value" style="color: var(--success);">{high}</div>
                    <div class="stat-label">🟢 High</div>
                </div>
                ''', unsafe_allow_html=True)
            with c3:
                st.markdown(f'''
                <div class="stat-card" style="animation-delay: 0.19s;">
                    <div class="stat-value" style="color: var(--warning);">{medium}</div>
                    <div class="stat-label">🟡 Medium</div>
                </div>
                ''', unsafe_allow_html=True)
            with c4:
                st.markdown(f'''
                <div class="stat-card" style="animation-delay: 0.26s;">
                    <div class="stat-value" style="color: var(--destructive);">{low}</div>
                    <div class="stat-label">🔴 Low</div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.divider()
            
            # Show ALL questions (no limit)
            st.markdown(f'**📋 All {len(ranked)} Questions (Ranked by Score)**')
            
            for i, result in enumerate(ranked):
                q = result['question']
                pred = result['prediction']
                score = result['score']
                score10 = round(score, 1)
                original = questions[cleaned_questions.index(q)] if q in cleaned_questions else q
    # Strip any leading "N." line number artifacts from the file
                import re
                clean_original = re.sub(r'^\d+[\.\)]\s*', '', original)
    
                if score10 >= 7:
                    badge_color = "var(--success)"
                    badge_label = "High"
                elif score10 >= 4:
                    badge_color = "var(--warning)"
                    badge_label = "Medium"
                else:
                    badge_color = "var(--destructive)"
                    badge_label = "Low"

                with st.expander(f"#{i+1}  ·  {score10}/10  ·  {clean_original[:65]}"):
                    st.markdown(f"""
                    <div style="padding: 8px 0;">
                        <p style="color: var(--text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;">Full Question</p>
                        <p style="color: var(--text-primary); font-size: 15px; margin-bottom: 16px;">{clean_original}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(score, 10.0) / 10.0)
                    st.markdown(f"""
                    <div style="margin-top: 12px;">
                        <span style="background: transparent; border: 1px solid {badge_color}; color: {badge_color};
                         padding: 6px 16px; border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">
                               {badge_label} Quality
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            csv_data = "Question,Score,Quality\n"
            for result in ranked:
                q = result['question']
                s = result['score']
                original = questions[cleaned_questions.index(q)] if q in cleaned_questions else q
                quality = "High" if s >= 7 else "Medium" if s >= 4 else "Low"
                csv_data += f'"{original}",{s:.1f},{quality}\n'
            
            # ✅ Export CSV button now matches (black background, white text)
            st.download_button(
                label="📥 Export All as CSV",
                data=csv_data,
                file_name="loqqin-batch.csv",
                use_container_width=True
            )
            
            if st.button("🗑️ Clear & Start Over", use_container_width=True, type="secondary"):
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)