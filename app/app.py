# -----------------------------
# IMPORTS
# -----------------------------
import sys
import os
# ✅ FIXED: __file__ with double underscores
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
# ENHANCED GLASSMORPHISM CSS
# Fixed visibility + smoother animations
# -----------------------------
st.markdown("""
<style>
    /* Import Inter Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* CSS Variables */
    :root {
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --glass-bg: rgba(255, 255, 255, 0.95);
        --glass-border: rgba(255, 255, 255, 0.4);
        --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        --blur: 16px;
        
        --foreground: #1A1A2E;
        --foreground-secondary: #4A4A68;
        --muted-foreground: #6B7280;
        --primary: #6366F1;
        --primary-hover: #4F46E5;
        --success: #16A34A;
        --warning: #D97706;
        --destructive: #DC2626;
        --radius: 18px;
    }

    /* Background with Gradient */
    .stApp {
        background: var(--background-gradient);
        background-attachment: fixed;
        min-height: 100vh;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Header with Better Contrast */
    .loqqin-header {
        text-align: center;
        margin-bottom: 35px;
        padding: 35px 0;
    }
    .loqqin-header h1 {
        font-size: 42px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        text-shadow: 0 4px 16px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }
    .loqqin-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 16px;
        margin: 12px 0 0 0;
        font-weight: 400;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    /* Glassmorphism Card - ENHANCED VISIBILITY */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(var(--blur));
        -webkit-backdrop-filter: blur(var(--blur));
        border: 1px solid var(--glass-border);
        border-radius: var(--radius);
        padding: 32px;
        margin: 20px 0;
        box-shadow: var(--glass-shadow);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.98);
        box-shadow: 0 16px 48px 0 rgba(31, 38, 135, 0.3);
        transform: translateY(-3px);
    }

    /* Button with Better Contrast */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%);
        color: white !important;
        border-radius: calc(var(--radius) - 6px);
        padding: 14px 32px;
        border: 1px solid rgba(255,255,255,0.3);
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        width: 100%;
        height: 52px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
        letter-spacing: 0.2px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(99, 102, 241, 0.5);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-hover) 100%);
        border-radius: 999px;
        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.5);
    }
    .stProgress > div {
        height: 16px;
        border-radius: 999px;
        background: rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(4px);
    }

    /* Score Display */
    .score-display {
        font-size: 56px;
        font-weight: 800;
        background: linear-gradient(135deg, var(--foreground) 0%, var(--primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 18px 0;
        text-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .score-display span {
        font-size: 24px;
        font-weight: 600;
        color: var(--muted-foreground);
        -webkit-text-fill-color: var(--muted-foreground);
    }

    /* Glass Badges - ENHANCED */
    .badge {
        display: inline-block;
        padding: 10px 24px;
        border-radius: 999px;
        font-size: 15px;
        font-weight: 700;
        width: 100%;
        text-align: center;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .badge-high {
        background: rgba(22, 163, 74, 0.25);
        color: var(--success);
        border-color: rgba(22, 163, 74, 0.4);
    }
    .badge-med {
        background: rgba(217, 119, 6, 0.25);
        color: var(--warning);
        border-color: rgba(217, 119, 6, 0.4);
    }
    .badge-low {
        background: rgba(220, 38, 38, 0.25);
        color: var(--destructive);
        border-color: rgba(220, 38, 38, 0.4);
    }

    /* Glass Metrics Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 22px;
        border-radius: calc(var(--radius) - 6px);
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
        border: 1px solid rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.85);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    .metric-label {
        color: var(--muted-foreground);
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        color: var(--foreground);
        font-size: 21px;
        font-weight: 700;
    }

    /* Text Area - ENHANCED VISIBILITY */
    .stTextArea label {
        font-weight: 700;
        color: var(--foreground);
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        margin-bottom: 8px;
    }
    .stTextArea textarea {
        font-family: 'Inter', sans-serif;
        border-radius: calc(var(--radius) - 6px);
        border: 2px solid rgba(0,0,0,0.1);
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(4px);
        font-size: 15px;
        padding: 14px;
        color: var(--foreground);
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        outline: none;
    }

    /* Character Counter - FIXED VISIBILITY */
    .char-counter {
        color: rgba(255, 255, 255, 0.9);
        font-size: 13px;
        font-weight: 500;
        margin-top: 8px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(0,0,0,0.1);
        margin: 30px 0;
    }

    /* Tabs - ENHANCED */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.15);
        padding: 8px;
        border-radius: calc(var(--radius) + 6px);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: calc(var(--radius) - 6px);
        background: transparent;
        border: none;
        color: rgba(255,255,255,0.8);
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.2);
        color: white;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white;
        color: var(--primary);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        font-weight: 700;
    }

    /* Subheader & Caption - FIXED VISIBILITY */
    .stSubheader {
        color: var(--foreground);
        font-weight: 700;
        font-size: 22px;
        margin-bottom: 8px;
    }
    .stCaption {
        color: rgba(255, 255, 255, 0.9);
        font-size: 14px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Expander (Glass) */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(8px);
        border-radius: calc(var(--radius) - 6px);
        border: 1px solid rgba(0,0,0,0.1);
    }

    /* Success/Warning/Error Messages */
    .stAlert {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
        border-radius: calc(var(--radius) - 6px);
        border: 1px solid rgba(0,0,0,0.1);
    }

    /* Smooth Scroll */
    html {
        scroll-behavior: smooth;
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .glass-card {
        animation: fadeInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="loqqin-header">
    <h1>🧠 LOQQIN</h1>
    <p>Learning-Oriented Question Quality Predictor</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["🔍 Single Analyzer", "📦 Batch Upload"])

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
    
    # ✅ FIXED: Character counter with better visibility
    st.markdown(f'<p class="char-counter">📝 {len(question)} characters</p>', unsafe_allow_html=True)
    
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
            st.markdown('<div style="text-align: center; margin: 40px 0;">', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">✨ Quality Score</p>', unsafe_allow_html=True)
            st.progress(min(score, 10.0) / 10.0)
            st.markdown(f'<div class="score-display">{score10}<span>/10</span></div>', unsafe_allow_html=True)
            
            # Badge
            if score10 >= 7:
                badge_class = "badge-high"
                badge_text = "🟢 High Quality Question ⭐"
            elif score10 >= 4:
                badge_class = "badge-med"
                badge_text = "🟡 Medium Depth Question"
            else:
                badge_class = "badge-low"
                badge_text = "🔴 Surface Level Question"
            
            st.markdown(f'<div class="badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Metrics Grid
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">👁️ Clarity</div>
                    <div class="metric-value">{clarity}</div>
                </div>
                ''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">🎯 Specificity</div>
                    <div class="metric-value">{specificity}</div>
                </div>
                ''', unsafe_allow_html=True)
            with c3:
                st.markdown(f'''
                <div class="metric-card">
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
            high = sum(1 for _, _, s in ranked if s >= 7)
            medium = sum(1 for _, _, s in ranked if 4 <= s < 7)
            low = sum(1 for _, _, s in ranked if s < 4)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📋 Total", len(questions))
            c2.metric("🟢 High", high)
            c3.metric("🟡 Medium", medium)
            c4.metric("🔴 Low", low)
            
            st.divider()
            
            # Question List
            for i, (q, pred, score) in enumerate(ranked[:10]):
                score10 = round(score, 1)
                original = questions[cleaned_questions.index(q)] if q in cleaned_questions else q
                
                with st.expander(f"**#{i+1}** | {score10}/10 | {original[:55]}..."):
                    st.write(f"**Full Question:** {original}")
                    st.progress(min(score, 10.0) / 10.0)
                    
                    if score10 >= 7:
                        st.success("🟢 High Quality ⭐")
                    elif score10 >= 4:
                        st.info("🟡 Medium Quality")
                    else:
                        st.warning("🔴 Low Quality")
            
            st.divider()
            csv_data = "Question,Score,Quality\n"
            for q, _, s in ranked:
                original = questions[cleaned_questions.index(q)] if q in cleaned_questions else q
                quality = "High" if s >= 7 else "Medium" if s >= 4 else "Low"
                csv_data += f'"{original}",{s:.1f},{quality}\n'
            
            st.download_button(
                label="📥 Export All as CSV",
                data=csv_data,
                file_name="loqqin-batch.csv",
                use_container_width=True
            )
            
            if st.button("🗑️ Clear & Start Over", use_container_width=True, type="secondary"):
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

