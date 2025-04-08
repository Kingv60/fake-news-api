import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load the saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download stopwords (only needed once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to predict news category
def predict_news(news_text):
    cleaned_text = clean_text(news_text)  # Preprocess text
    text_tfidf = vectorizer.transform([cleaned_text])  # Convert to TF-IDF
    prediction = model.predict(text_tfidf)[0]  # Predict using model
    return "Fake News" if prediction == 1 else "Real News"

# Example usage
news_example = input("Enter Your News : ")
result = predict_news(news_example)
print("Prediction:", result)
