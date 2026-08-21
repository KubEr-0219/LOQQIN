"""Text preprocessing for LOQQIN."""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def _ensure_nltk_resources() -> None:
    """Download NLTK resources only when they are missing."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_resources()
_STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Normalize question text for TF-IDF features."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = word_tokenize(text)
    return " ".join(word for word in words if word not in _STOP_WORDS)
