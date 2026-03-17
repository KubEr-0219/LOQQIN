from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(X, y):
    base = LinearSVC(max_iter=2000)
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X, y)
    return model


# -----------------------------
# RULE BASED SCORING
# -----------------------------
def rule_based_score(question):

    question = question.lower()
    score = 0

    # Deep / conceptual indicators
    deep_keywords = [
        "explain",
        "compare",
        "analyze",
        "architecture",
        "design",
        "discuss",
        "why",
        "how",
        "evaluate"
    ]

    # Surface / basic indicators
    surface_keywords = [
        "define",
        "what is",
        "list",
        "name",
        "state",
        "identify"
    ]

    for word in deep_keywords:
        if word in question:
            score += 2

    for word in surface_keywords:
        if word in question:
            score -= 2

    return score


# -----------------------------
# PREDICT SINGLE QUESTION
# -----------------------------
def predict_question(model, vectorizer, question):

    question_vector = vectorizer.transform([question])

    prediction = model.predict(question_vector)[0]

    # Real probability instead of arbitrary decision_function
    proba = model.predict_proba(question_vector)[0]
    confidence = proba[1]  # probability of the predicted class (0.0 - 1.0)

    # Scale to 0-10
    ml_score = confidence * 10

    # Rule score (kept as modifier, reduced weight)
    rule_score = rule_based_score(question)
    rule_modifier = max(min(rule_score * 0.5, 2), -2)  # cap rule influence at ±2

    final_score = ml_score + rule_modifier

    # Clamp to 0–10
    final_score = max(min(final_score, 10), 0)

    return prediction, final_score

# -----------------------------
# RANK MULTIPLE QUESTIONS
# -----------------------------
def rank_questions(model, vectorizer, questions):

    results = []

    for q in questions:
        prediction, score = predict_question(model, vectorizer, q)
        results.append((q, prediction, score))

    # Sort highest score first
    results.sort(key=lambda x: x[1], reverse=True)

    return results

# -----------------------------
# ANALYZE METRICS
# -----------------------------
def analyze_question_metrics(question):
    q = question.lower()

    # --- BLOOM'S LEVEL ---
    bloom_map = {
        "Remember":     ["define", "list", "name", "state", "recall", "identify", "what is", "when", "who"],
        "Understand":   ["explain", "describe", "summarize", "interpret", "classify", "what are"],
        "Apply":        ["solve", "use", "demonstrate", "calculate", "apply", "implement"],
        "Analyze":      ["analyze", "compare", "contrast", "differentiate", "examine", "why", "how does"],
        "Evaluate":     ["evaluate", "justify", "assess", "critique", "argue", "defend"],
        "Create":       ["design", "develop", "construct", "propose", "formulate", "create", "build"]
    }

    bloom_level = "Remember"  # default
    for level, keywords in bloom_map.items():
        if any(kw in q for kw in keywords):
            bloom_level = level

    # --- CLARITY ---
    word_count = len(q.split())
    if word_count < 4:
        clarity = "Low"
    elif word_count <= 15:
        clarity = "High"
    else:
        clarity = "Medium"

    # --- SPECIFICITY ---
    vague_terms = ["something", "things", "stuff", "etc", "and so on", "various"]
    specific_indicators = ["architecture", "algorithm", "protocol", "mechanism", "process", "system", "model", "network", "structure"]

    vague_count = sum(1 for term in vague_terms if term in q)
    specific_count = sum(1 for term in specific_indicators if term in q)

    if specific_count >= 2:
        specificity = "High"
    elif vague_count >= 1:
        specificity = "Low"
    else:
        specificity = "Medium"

    return clarity, specificity, bloom_level