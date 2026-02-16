import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# download once (first run only)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # lowercase
    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenize
    words = word_tokenize(text)

    # remove stopwords
    filtered_words = [w for w in words if w not in stop_words]

    # join back to sentence
    cleaned_text = " ".join(filtered_words)

    return cleaned_text
