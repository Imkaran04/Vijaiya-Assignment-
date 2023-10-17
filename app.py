import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('model/random_forest_model.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.joblib')

# Define the list of class labels in the order they were encoded during training
class_labels = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text', "")

    # Check if the length is between 2 and 150 characters
    if not (2 <= len(text) <= 150):
        return jsonify({"error": "Text length must be between 2 and 150 characters."}), 400

    # Vectorize the input text using the TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([text])

    # Get model prediction
    predicted_label = model.predict(vectorized_text)[0]


    return jsonify({"classification": predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
