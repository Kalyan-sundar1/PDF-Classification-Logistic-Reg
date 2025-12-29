import os
import pdfplumber
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# -----------------------------
# PDF TEXT EXTRACTION FUNCTION
# -----------------------------
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# -----------------------------
# DATA CONFIG
# -----------------------------
BASE_DIR = "data"

label_map = {
    "resume": 0,
    "invoice": 1,
    "article": 2,
    "annual_report": 3,
    "contract": 4,
    "receipt": 5
}


# -----------------------------
# LOAD DATA
# -----------------------------
texts = []
labels = []

for folder, label in label_map.items():
    folder_path = os.path.join(BASE_DIR, folder)

    if not os.path.exists(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            texts.append(extract_text(pdf_path))
            labels.append(label)

df = pd.DataFrame({"text": texts, "label": labels})

print("📊 Class distribution:")
print(df["label"].value_counts())


# -----------------------------
# VECTORIZE TEXT
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression(
    max_iter=2000,
    solver="saga"
)

model.fit(X_train, y_train)


# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# CLASSIFICATION REPORT (FIXED)
# -----------------------------
labels_present = np.unique(y_test)
target_names = [k for k, v in label_map.items() if v in labels_present]

print("\n📄 Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=labels_present,
    target_names=target_names,
    zero_division=0
))


# -----------------------------
# SAVE MODEL & VECTORIZER
# -----------------------------
joblib.dump(model, "document_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully")
