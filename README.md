# 🧠 LOQQIN
### Learning-Oriented Question Quality Predictor

> An ML-powered web application that automatically evaluates the quality of exam questions, classifies them by Bloom's Taxonomy level, and ranks entire question papers — instantly.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [ML Pipeline](#ml-pipeline)
- [Scoring System](#scoring-system)
- [Bloom's Taxonomy Mapping](#blooms-taxonomy-mapping)
- [Dataset](#dataset)
- [Team](#team)

---

## 📖 Overview

In educational settings, question quality varies significantly — some questions test surface-level memorization while others assess deep conceptual understanding. Manually reviewing large question banks is time-consuming, subjective, and inconsistent.

**LOQQIN** solves this by providing an automated, deterministic quality scoring pipeline. Given any exam question, LOQQIN returns:

- A **quality score** from 0 to 10
- A **Bloom's Taxonomy level** (Remember → Create)
- **Clarity** and **Specificity** metrics
- A **quality badge** (High / Medium / Low)

For educators managing large question banks, the **batch upload** feature ranks an entire question paper by quality in seconds.

---

## ✨ Features

| Feature | Description |
|---|---|
| Single Question Analyzer | Paste any question and get instant quality feedback |
| Bloom's Level Detection | Automatically maps to all 6 Bloom's Taxonomy levels |
| Clarity & Specificity Metrics | Word-count and domain-term based analysis |
| Batch Upload & Ranking | Upload a .txt file, get all questions ranked by score |
| Quality Stats Dashboard | High / Medium / Low breakdown for batch results |
| Export Results | Download single result as .txt or batch results as .csv |
| Custom Dark UI | MerchBanao-inspired dark design system with Syne + DM Sans fonts |
| Print-Friendly | @media print CSS converts dark theme to clean white for printouts |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Custom CSS (Syne + DM Sans, CSS Variables) |
| ML Model | scikit-learn — StackingRegressor (Ridge + GBR + SVR) |
| Text Features | TF-IDF Vectorizer (unigrams + bigrams, 3000 features) |
| Preprocessing | NLTK — tokenization, stopword removal |
| Persistence | joblib — model.pkl, vectorizer.pkl |
| Data | pandas — questions.csv |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
LOQQIN/
│
├── app/
│   └── app.py                  # Streamlit frontend — UI, tabs, rendering
│
├── src/
│   ├── model.py                # ML model, scoring, metrics, ranking
│   ├── features.py             # TF-IDF vectorizer creation
│   ├── preprocess.py           # Text cleaning pipeline (NLTK)
│   └── utils.py                # Save/load model pkl files
│
├── data/
│   └── questions.csv           # Labeled training dataset
│
├── notebooks/                  # Exploratory notebooks (if any)
│
├── main.py                     # Training script — run this first
├── test.py                     # Diagnostic script — checks environment
├── model.pkl                   # Serialized trained model
├── vectorizer.pkl              # Serialized TF-IDF vectorizer
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/KubEr-0219/LOQQIN.git
cd LOQQIN
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download NLTK data** (first time only)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**5. Train the model**
```bash
python main.py
```
This generates `model.pkl` and `vectorizer.pkl` in the project root.

**6. Run the app**
```bash
cd app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🚀 Usage

### Single Question Analysis
1. Open the app in your browser
2. Go to the **Single Analyzer** tab
3. Type or paste your question into the text area
4. Click **✨ Analyze Question**
5. View your score, quality badge, and Bloom's metrics
6. Optionally export the result as a `.txt` report

### Batch Upload
1. Prepare a `.txt` file with one question per line
2. Go to the **Batch Upload** tab
3. Upload your file
4. Click **✨ Analyze Questions**
5. View ranked results with stats breakdown
6. Export all results as `.csv`

### Diagnostics
If the app fails to load, run the diagnostic script:
```bash
python test.py
```
This checks file presence, model loading, and runs a test prediction.

---

## 🤖 ML Pipeline

```
Raw Question
     │
     ▼
preprocess.py ──── lowercase → remove punctuation → tokenize → remove stopwords
     │
     ▼
features.py ────── TF-IDF vectorization (unigrams + bigrams, max 3000 features)
     │
     ▼
model.py ──────── StackingRegressor
                   ├── Ridge (alpha=1.0)
                   ├── GradientBoostingRegressor
                   ├── SVR (kernel='rbf')
                   └── Meta-learner: Ridge (alpha=0.5)
     │
     ▼
rule_based_score() ── keyword modifier (±2 cap)
     │
     ▼
Final Score (0–10) + Bloom's Level + Clarity + Specificity
```

---

## 📊 Scoring System

LOQQIN uses a **hybrid scoring system**:

```
ml_score      = StackingRegressor prediction (0–10)
rule_modifier = clamp(keyword_score × 0.5, -2, +2)
final_score   = clamp(ml_score + rule_modifier, 0, 10)
```

| Score Range | Quality Badge |
|---|---|
| 7.0 – 10.0 | 🟢 High Quality |
| 4.0 – 6.9 | 🟡 Medium Depth |
| 0.0 – 3.9 | 🔴 Surface Level |

### Rule-Based Keyword Modifiers

| Keywords | Effect |
|---|---|
| design, evaluate, analyze, compare | Positive modifier |
| define, list, name, state, identify | Negative modifier |

---

## 📚 Bloom's Taxonomy Mapping

LOQQIN detects Bloom's level hierarchically from highest to lowest:

| Level | Trigger Keywords |
|---|---|
| **Create** | design, develop, construct, propose, build, architect |
| **Evaluate** | evaluate, justify, assess, critique, argue, defend |
| **Analyze** | analyze, compare, contrast, differentiate, examine, why |
| **Apply** | solve, use, demonstrate, calculate, apply, implement |
| **Understand** | explain, describe, summarize, interpret, classify |
| **Remember** | define, list, name, state, recall, identify, what is |

---

## 📂 Dataset

The training dataset (`data/questions.csv`) contains labeled exam questions:

| Column | Description |
|---|---|
| `question` | Raw question text |
| `label` | 1 = High quality / Deep, 0 = Surface level |

Questions were manually curated and labeled based on Bloom's Taxonomy principles across multiple engineering subjects including IoT, Machine Learning, and Computer Science.

---

## 📄 License

This project was developed as an academic project at Swami Vivekananda Institute of Technology.

---

<div align="center">
  <p>Built with ❤️ at SVIT · LOQQIN © 2026</p>
</div>
