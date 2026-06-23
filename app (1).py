
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

app = Flask(__name__)

# Global variables to hold the trained vectorizer and classifier
vectorizer = None
clf = None

def train_model():
    global vectorizer, clf
    # Load the dataset (assuming train.csv is in the same directory)
    if not os.path.exists('train.csv'):
        print("Error: train.csv not found. Please ensure it's in the same directory as app.py")
        exit(1)

    df = pd.read_csv('train.csv')

    # Prepare data
    message = 'sms'
    target = 'label'
    X = df[message]
    y = df[target]

    # Split data (necessary for fitting the vectorizer and classifier)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text
    vectorizer = CountVectorizer(stop_words='english')
    X_train_counts = vectorizer.fit_transform(X_train)

    # Train the Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)
    print("Model trained successfully.")


@app.route('/predict', methods=['POST'])
def predict_spam_or_ham_api():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    sms_message = data.get('message', '')

    if not sms_message.strip():
        return jsonify({"prediction": "Please provide a message for prediction."}), 400

    if vectorizer is None or clf is None:
        return jsonify({"error": "Model not trained yet. Please wait for initialization or check for errors."}), 500

    # Transform the input message
    input_counts = vectorizer.transform([sms_message])

    # Predict
    prediction_label = clf.predict(input_counts)[0]

    result = "HAM MESSAGE" if prediction_label == 0 else "SPAM MESSAGE"
    return jsonify({"prediction": result})


@app.route('/')
def home():
    return "SMS Spam Detection API. Use /predict endpoint with a POST request."


if __name__ == '__main__':
    # Train the model when the application starts
    train_model()
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
