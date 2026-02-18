import joblib

def save_objects(model, vectorizer):
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")


def load_objects():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer
