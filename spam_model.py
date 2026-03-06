# SMS Spam Detection - Final Version

import pandas as pd
import string
import nltk
import numpy as np

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords (only first time)
nltk.download('stopwords')


# 1. Load Dataset


df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# 2. Text Preprocessing

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['message'] = df['message'].apply(preprocess)


# 3. Feature Extraction (TF-IDF)


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['message'])
y = df['label']


# 4. Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 5. Model 1 - Naive Bayes


nb_model = MultinomialNB(alpha=0.5)
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("\n====== NAIVE BAYES RESULTS ======")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))


# 6. Model 2 - Random Forest


rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n====== RANDOM FOREST RESULTS ======")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))


# 7. Custom Prediction Function


def predict_message(msg, model):
    msg = preprocess(msg)
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    return "SPAM" if prediction == 1 else "HAM"


# 8. Sample Predictions


sample_msgs = [
    "Congratulations! You won a free prize",
    "Hey bro, are we meeting today?",
    "Free entry in a weekly lottery",
    "Call me when you are free"
]

print("\n====== SAMPLE PREDICTIONS ======")
for msg in sample_msgs:
    print(f"\nMessage: {msg}")
    print("Naive Bayes Prediction:", predict_message(msg, nb_model))
    print("Random Forest Prediction:", predict_message(msg, rf_model))


# 9. Interactive Input


print("\n====== INTERACTIVE SMS SPAM PREDICTION ======")
print("Type 'exit' to quit.")

while True:
    msg = input("\nEnter SMS message: ")
    if msg.lower() == 'exit':
        print("Exiting...")
        break
    print("Naive Bayes Prediction:", predict_message(msg, nb_model))
    print("Random Forest Prediction:", predict_message(msg, rf_model))