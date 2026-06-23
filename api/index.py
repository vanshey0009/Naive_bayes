import os
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define paths for loading the model and vectorizer relative to the API file
# Assuming model.pkl and vectorizer.pkl are in the parent directory of api/
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'vectorizer.pkl')

# Load the trained classifier model and vectorizer
try:
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    clf = None
    vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if clf is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not loaded'}), 500

    if not request.json or 'message' not in request.json:
        return jsonify({'error': 'Please provide a message in JSON format with a 'message' key'}), 400

    message = request.json['message']

    if not message or not isinstance(message, str):
        return jsonify({'error': 'Message must be a non-empty string'}), 400

    # Preprocess the message
    message_vectorized = vectorizer.transform([message])

    # Make prediction
    prediction = clf.predict(message_vectorized)[0]

    # Return result
    result = 'spam' if prediction == 1 else 'ham'
    return jsonify({'message': message, 'prediction': result})

@app.route('/', methods=['GET'])
def home():
    return "Naive Bayes Spam Detector API. Use /predict with POST request."


# To run the app locally during development (not for Vercel deployment)
# if __name__ == '__main__':
#     app.run(debug=True)
