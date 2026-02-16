from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(text_data):
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(text_data)

    return X, vectorizer
