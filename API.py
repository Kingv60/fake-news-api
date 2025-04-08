from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download stopwords (first-time use)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Prediction endpoint
@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({"response": "No text provided."}), 400

    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]

    result = "Fake News" if prediction == 1 else "Real News"
    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

