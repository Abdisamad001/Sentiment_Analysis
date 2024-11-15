import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from gensim.models import Word2Vec
import os
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def balance_dataset(X, y):
    # Combine features and labels
    data = np.hstack((X, y.reshape(-1,1)))
    
    # Separate by class
    classes = []
    for i in range(5):  # 5 classes (0-4)
        class_data = data[data[:,-1] == i]
        classes.append(class_data)
    
    # Find class with minimum samples
    min_samples = min(len(c) for c in classes)
    
    # Downsample each class
    balanced_classes = []
    for c in classes:
        downsampled = resample(c, 
                             n_samples=min_samples,
                             random_state=42)
        balanced_classes.append(downsampled)
    
    # Combine balanced data
    balanced_data = np.vstack(balanced_classes)
    
    # Separate features and labels again
    X_balanced = balanced_data[:,:-1]
    y_balanced = balanced_data[:,-1]
    
    return X_balanced, y_balanced

def preprocess_text(text):
    text = str(text).lower()
    # Additional sentiment indicators
    exclamation_count = text.count('!')
    question_count = text.count('?')
    upper_case_count = sum(1 for c in str(text) if c.isupper())
    
    # Common sentiment words counts
    positive_words = ['good', 'great', 'awesome', 'excellent', 'love']
    negative_words = ['bad', 'poor', 'terrible', 'hate', 'worst']
    positive_count = sum(text.count(word) for word in positive_words)
    negative_count = sum(text.count(word) for word in negative_words)
    
    # Clean text
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens), exclamation_count, question_count, upper_case_count, positive_count, negative_count

def get_document_vector(text, model):
    words = text.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)

# Main execution
print("Loading data...")
df = pd.read_csv('data/raw/all_kindle_review.csv')

# Take a smaller subset of the data to reduce memory usage
df = df.sample(n=min(len(df), 50000), random_state=42)

processed_data = [preprocess_text(text) for text in df['reviewText'].fillna('')]
df['processed_text'] = [text[0] for text in processed_data]
df['exclamation_count'] = [text[1] for text in processed_data]
df['question_count'] = [text[2] for text in processed_data]
df['upper_case_count'] = [text[3] for text in processed_data]
df['positive_count'] = [text[4] for text in processed_data]
df['negative_count'] = [text[5] for text in processed_data]

print("Training Word2Vec model...")
tokenized_texts = [text.split() for text in df['processed_text']]
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=50, window=5, min_count=1, workers=4)

print("Creating features...")
tfidf = TfidfVectorizer(max_features=500)
tfidf_features = tfidf.fit_transform(df['processed_text']).toarray()

w2v_features = np.array([get_document_vector(text, w2v_model) for text in df['processed_text']])

# Combine all features
extra_features = np.column_stack((
    df['exclamation_count'],
    df['question_count'],
    df['upper_case_count'],
    df['positive_count'],
    df['negative_count'],
    df['processed_text'].str.len(),
    df['processed_text'].str.split().str.len()
))

X = np.hstack((tfidf_features, w2v_features, extra_features))
y = df['rating'].values - 1

print("Balancing dataset...")
X_balanced, y_balanced = balance_dataset(X, y)

print("Selecting features...")
selector = SelectKBest(f_classif, k=500)
X_selected = selector.fit_transform(X_balanced, y_balanced)

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_balanced, 
                                                    test_size=0.2, 
                                                    random_state=42)

print("Training models...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Save models and preprocessors
print("Saving models...")
models = {
    'rf_model': rf_model,
    'xgb_model': xgb_model,
    'voting_clf': voting_clf,
    'w2v_model': w2v_model,
    'tfidf': tfidf,
    'scaler': scaler,
    'selector': selector
}

os.makedirs('models', exist_ok=True)
with open('models/sentiment_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("Models saved successfully!")

print("\nRandom Forest Performance:")
rf_pred = rf_model.predict(X_test)
print(classification_report(y_test, rf_pred))

print("\nXGBoost Performance:")
xgb_pred = xgb_model.predict(X_test)
print(classification_report(y_test, xgb_pred))

print("\nVoting Classifier Performance:")
voting_pred = voting_clf.predict(X_test)
print(classification_report(y_test, voting_pred))