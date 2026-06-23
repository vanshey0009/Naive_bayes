
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys

# Load the dataset (assuming train.csv is in the same directory)
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

def predict_spam_or_ham(sms_message):
    if not sms_message.strip():
        return "Please enter a message."

    # Transform the input message
    input_counts = vectorizer.transform([sms_message])

    # Predict
    prediction = clf.predict(input_counts)[0]

    if prediction == 0:
        return "HAM MESSAGE"
    else:
        return "SPAM MESSAGE"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_message = " ".join(sys.argv[1:])
        print(predict_spam_or_ham(input_message))
    else:
        print("Usage: python spam_detector.py \"Your message here\"")
