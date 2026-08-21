"""Feature extraction configuration for LOQQIN."""

from sklearn.feature_extraction.text import TfidfVectorizer



def create_tfidf_features(text_data):
    """Fit the canonical TF-IDF vectorizer and transform text data."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=5000,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]+\b",
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer
