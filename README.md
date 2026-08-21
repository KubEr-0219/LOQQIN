# 🧠 LOQQIN
### Learning-Oriented Question Quality Predictor

LOQQIN is an NLP application that predicts the quality of examination questions on a **0–10 scale**, estimates Bloom's Taxonomy level, reports transparent clarity/specificity signals, and ranks batches of questions.

The project is deliberately split into two parts:

- **Learned ML score:** TF-IDF + StackingRegressor predicts the 0–10 quality score.
- **Interpretable educational signals:** Bloom's level, clarity, specificity, and an optional keyword heuristic are deterministic features and **do not modify the ML prediction**.

## Features

| Feature | Description |
|---|---|
| Single Analyzer | Score an individual exam/assessment question |
| Bloom's Taxonomy | Deterministic mapping from Remember → Create |
| Clarity & Specificity | Transparent rule-based explanatory metrics |
| Batch Ranking | Score and rank multiple questions |
| Export | Download individual and batch results |
| Streamlit UI | Interactive browser-based interface |
| Reproducible Evaluation | Leakage-safe 5-fold cross-validation and model comparison |
| Experiment Tracking | Saves evaluation results to `experiments/latest.json` |

## ML methodology

```text
Raw question
    ↓
Text normalization (lowercase, punctuation cleanup, tokenization)
    ↓
TF-IDF (1–3 grams, up to 5,000 features, sublinear TF)
    ↓
StackingRegressor
    ├── Ridge
    ├── GradientBoostingRegressor
    ├── SVR (RBF)
    └── Ridge meta-learner
    ↓
Predicted quality score (0–10)
```

The production model contains **no hardcoded keyword prediction overrides**. A question containing words such as `design`, `explain`, or `define` is still scored by the learned model rather than being forced to a fixed value.

### Separate classification benchmark

For model comparison, labels are also converted to a binary educational tier:

- `0–3` → lower-quality tier
- `4–10` → higher-quality tier

A separate Logistic Regression / Linear SVM benchmark reports Accuracy, F1, and ROC-AUC. These classification metrics are **not** used to claim regression accuracy for the 0–10 score.

## Evaluation methodology

Run:

```bash
python evaluate.py
```

The evaluation uses **5-fold shuffled cross-validation with a fixed random seed (42)**. TF-IDF is fitted **inside each fold**, preventing vocabulary/IDF information from leaking from validation folds into training folds.

Regression models compared:

- Ridge
- Gradient Boosting
- SVR
- StackingRegressor

Regression metrics:

- MAE — mean absolute error on the 0–10 scale
- RMSE — root mean squared error
- R² — explained variance relative to the dataset mean

Classification metrics:

- Accuracy
- F1
- ROC-AUC

The script writes a reproducible experiment report to:

```text
experiments/latest.json
```

**Resume rule:** only report metrics produced by `evaluate.py` on the current commit. Do not describe R² as accuracy and do not report training-set metrics as model performance.

## Current baseline

Before the refactor, an external 5-fold evaluation of the original StackingRegressor reported approximately **R² 0.555, RMSE 2.41, and MAE 1.90** on the 0–10 scale. Those numbers are retained as a historical baseline only; rerun `python evaluate.py` after the refactor before using final resume metrics.

## Dataset

`questions.csv` contains manually curated examination questions from engineering/technology domains with labels from **0 to 10**. The labels encode intended question depth/quality using Bloom's Taxonomy principles.

Because the dataset is relatively small, results should be described as performance on this curated dataset rather than as universal question-quality accuracy.

## Project structure

```text
LOQQIN/
├── app/
│   └── app.py                  # Streamlit interface
├── src/
│   ├── features.py             # Canonical TF-IDF configuration
│   ├── model.py                # Regression model + inference + educational signals
│   ├── preprocess.py           # Text normalization
│   └── utils.py                # Model artifact persistence
├── tests/
│   └── test_model.py           # Unit tests
├── evaluate.py                 # Leakage-safe CV + model comparison
├── train.py                    # Final model training on all labeled data
├── test.py                     # Local model smoke test
├── questions.csv               # Labeled dataset
├── model.pkl                   # Trained regression artifact
├── vectorizer.pkl              # Trained TF-IDF artifact
├── requirements.txt
└── .github/workflows/ci.yml    # Automated test workflow
```

## Setup

```bash
git clone https://github.com/KubEr-0219/LOQQIN.git
cd LOQQIN
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

NLTK resources are downloaded only when they are missing. You can also download them manually:

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

## Train

Benchmark first:

```bash
python evaluate.py
```

Then fit the selected production architecture on the complete labeled dataset:

```bash
python train.py
```

Training saves synchronized copies of the model and vectorizer for both the repository root and Streamlit app.

## Run the application

```bash
cd app
streamlit run app.py
```

## Smoke test

```bash
python test.py
```

## Engineering notes

The project intentionally separates **prediction** from **explanation**. Bloom's Taxonomy, clarity, specificity, and the optional keyword heuristic are useful educational context, but they are not silently mixed into the ML score. This keeps the evaluation reproducible and makes the model behavior defensible in an interview.

## License

This project was developed as an academic project at Swami Vivekananda Institute of Technology.

<div align="center">Built with ❤️ at SVIT · LOQQIN © 2026</div>
