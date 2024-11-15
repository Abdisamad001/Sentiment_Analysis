from flask import Flask, render_template, request, jsonify
from src.components.model_test import predict_sentiment
import numpy as np

app = Flask(__name__)

def get_sentiment_label(predictions):
    # Calculate average prediction
    avg_prediction = float(np.mean([
        predictions['Random Forest Prediction'],
        predictions['XGBoost Prediction'],
        predictions['Voting Classifier Prediction']
    ]))
    
    # Define sentiment ranges
    if avg_prediction >= 4.5:
        return "very_positive", 5
    elif avg_prediction >= 3.5:
        return "positive", 4
    elif avg_prediction >= 2.5:
        return "neutral", 3
    elif avg_prediction >= 1.5:
        return "negative", 2
    else:
        return "very_negative", 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        review = data.get('review', '')
        
        if not review:
            return jsonify({'error': 'No review text provided'})

        # Get model predictions
        predictions = predict_sentiment(review)
        
        # Get sentiment label and rating
        sentiment_label, rating = get_sentiment_label(predictions)
        
        # Format predictions for response and convert numpy types to Python native types
        formatted_predictions = {
            'random_forest': float(predictions['Random Forest Prediction']),
            'xgboost': float(predictions['XGBoost Prediction']),
            'voting_classifier': float(predictions['Voting Classifier Prediction'])
        }

        return jsonify({
            'sentiment': sentiment_label,
            'rating': float(rating),
            'all_predictions': formatted_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)