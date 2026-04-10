import nltk
import string
import os
from nltk.corpus import stopwords

# Create local nltk_data directory for Streamlit Cloud
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Download to local directory if not present
try:
    nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

try:
    nltk.data.find('corpora/stopwords', paths=[nltk_data_dir])
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

# Add to path
nltk.data.path.insert(0, nltk_data_dir)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean text without relying on word_tokenize (avoids punkt issues)"""
    text = text.lower()
    
    # Remove punctuation manually
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Simple split (faster and doesn't require punkt)
    words = text.split()
    
    # Remove stopwords
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
    
    return " ".join(filtered_words)