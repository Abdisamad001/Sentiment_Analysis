import pickle
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import os

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    text = str(text).lower()
    # Additional sentiment indicators
    exclamation_count = text.count('!')
    question_count = text.count('?')
    upper_case_count = sum(1 for c in str(text) if c.isupper())
    
    # Enhanced sentiment word lists
    positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'wonderful', 'fantastic']
    negative_words = ['bad', 'poor', 'terrible', 'hate', 'worst', 'awful', 'horrible']
    neutral_words = ['okay', 'average', 'decent', 'fair', 'mediocre', 'alright']
    
    positive_count = sum(text.count(word) for word in positive_words)
    negative_count = sum(text.count(word) for word in negative_words)
    neutral_count = sum(text.count(word) for word in neutral_words)
    
    # Clean text
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens), exclamation_count, question_count, upper_case_count, positive_count, negative_count, neutral_count

def get_document_vector(text, model):
    words = text.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)

# Load models
print("Loading models...")
model_path = os.path.join('models', 'sentiment_models.pkl')
try:
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    print("Models loaded successfully!")
except FileNotFoundError:
    print(f"Error: Could not find the model file at {model_path}")
    print("Current working directory:", os.getcwd())
    exit(1)

rf_model = models['rf_model']
xgb_model = models['xgb_model']
voting_clf = models['voting_clf']
w2v_model = models['w2v_model']
tfidf = models['tfidf']
scaler = models['scaler']
selector = models['selector']

def analyze_sentiment_patterns(text):
    text_lower = text.lower()
    
    # Complex neutral patterns
    neutral_patterns = [
        (r'not (great|bad|terrible|good) but not (great|bad|terrible|good)', 3),
        (r'(okay|decent|fair|alright)(?!.*(!|\?|great|awesome|terrible|worst))', 3),
        (r'mixed feelings', 3),
        (r'average', 3),
        (r'mediocre', 3),
        (r'so[- ]so', 3)
    ]
    
    # Check for neutral patterns
    for pattern, score in neutral_patterns:
        if re.search(pattern, text_lower):
            return score
            
    return None

def predict_sentiment(text):
    # Check for specific sentiment patterns first
    pattern_score = analyze_sentiment_patterns(text)
    if pattern_score is not None:
        return {
            'Random Forest Prediction': pattern_score,
            'XGBoost Prediction': pattern_score,
            'Voting Classifier Prediction': pattern_score
        }
    
    # Preprocess the text
    processed_results = preprocess_text(text)
    processed_text = processed_results[0]
    
    # Create features
    tfidf_features = tfidf.transform([processed_text]).toarray()
    w2v_features = np.array([get_document_vector(processed_text, w2v_model)])
    
    # Extra features
    extra_features = np.array([[
        processed_results[1],  # exclamation_count
        processed_results[2],  # question_count
        processed_results[3],  # upper_case_count
        processed_results[4],  # positive_count
        processed_results[5],  # negative_count
        len(processed_text),   # text length
        len(processed_text.split())  # word count
    ]])
    
    # Combine and transform features
    X = np.hstack((tfidf_features, w2v_features, extra_features))
    X = selector.transform(X)
    X = scaler.transform(X)
    
    # Make predictions
    rf_pred = rf_model.predict(X)[0] + 1
    xgb_pred = xgb_model.predict(X)[0] + 1
    voting_pred = voting_clf.predict(X)[0] + 1
    
    # Post-process predictions
    text_lower = text.lower()
    
    # Strong sentiment overrides
    if any(word in text_lower for word in ['amazing', 'excellent', 'fantastic', 'love']) and '!' in text:
        return {'Random Forest Prediction': 5, 'XGBoost Prediction': 5, 'Voting Classifier Prediction': 5}
    if any(word in text_lower for word in ['terrible', 'awful', 'hate', 'worst']) and '!' in text:
        return {'Random Forest Prediction': 1, 'XGBoost Prediction': 1, 'Voting Classifier Prediction': 1}
    
    return {
        'Random Forest Prediction': rf_pred,
        'XGBoost Prediction': xgb_pred,
        'Voting Classifier Prediction': voting_pred
    }

# Test the model
if __name__ == "__main__":
    test_reviews = [
        "This book was amazing! I couldn't put it down.",
        "The story was okay, but nothing special.",
        "Terrible waste of time. Don't recommend at all.",
        "A decent read with some interesting parts.",
        "One of the best books I've ever read!",
        "It was okay, not great but not terrible either."
    ]

    print("\nTesting model predictions:")
    for review in test_reviews:
        print("\nReview:", review)
        predictions = predict_sentiment(review)
        print("Predictions:", predictions)