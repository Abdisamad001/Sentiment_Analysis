import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        nltk.download(resource)

def preprocess_text(text):
    """Preprocess and tokenize text."""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'never'}
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens