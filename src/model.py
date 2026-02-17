from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    model = LogisticRegression()

    # train model
    model.fit(X, y)

    return model

def predict_question(model, vectorizer, question):
    # clean + transform question
    question_vector = vectorizer.transform([question])

    # prediction
    prediction = model.predict(question_vector)[0]

    # probability score
    confidence = model.predict_proba(question_vector)[0].max()

    return prediction, confidence

def rank_questions(model, vectorizer, questions):
    # convert questions to vectors
    vectors = vectorizer.transform(questions)

    # get probability scores
    probabilities = model.predict_proba(vectors)[:, 1]

    # combine questions with scores
    ranked = list(zip(questions, probabilities))

    # sort highest score first
    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked
