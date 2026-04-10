from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(text_data):
    # Enhanced vectorizer matching the improved model
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        ngram_range=(1, 3),  
        max_features=5000,   # Increased from 3000
        lowercase=True,
        token_pattern=r'(?u)\b[a-zA-Z]+\b',
        min_df=1,            # Include terms that appear at least once
        sublinear_tf=True    # Apply sublinear tf scaling
    )
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer