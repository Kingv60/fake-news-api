import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
fake_news = pd.read_csv(r"C:\Users\visha\Desktop\archive (1)\fake.csv")
real_news = pd.read_csv(r"C:\Users\visha\Desktop\archive (1)\true.csv")

# Add labels: 1 = Fake, 0 = Real
fake_news["label"] = 1
real_news["label"] = 0

# Combine both datasets
df = pd.concat([fake_news, real_news])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df["text"] = df["title"] + " " + df["text"]  # Combine title & text
df["text"] = df["text"].apply(clean_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Model and vectorizer saved successfully.")
