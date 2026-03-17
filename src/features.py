from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(text_data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2),   # captures phrases
        max_features=3000
    )

    X = vectorizer.fit_transform(text_data)

    return X, vectorizer