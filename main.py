import os
import re
import joblib
import pdfplumber
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix



# PDF TEXT EXTRACTION

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text



# TEXT CLEANING & NORMALIZATION

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



# DATA CONFIG

BASE_DIR = "data"

label_map = {
    "resume": 0,
    "invoice": 1,
    "article": 2,
    "annual_report": 3,
    "contract": 4,
    "receipt": 5
}



# LOAD DATA

texts = []
labels = []

for folder, label in label_map.items():
    folder_path = os.path.join(BASE_DIR, folder)

    if not os.path.exists(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            raw_text = extract_text(pdf_path)
            cleaned_text = clean_text(raw_text)

            texts.append(cleaned_text)
            labels.append(label)

df = pd.DataFrame({"text": texts, "label": labels})

print("\n📊 Original class distribution:")
print(df["label"].value_counts())



# FIX 1: REMOVE RARE CLASSES

MIN_SAMPLES = 2

class_counts = df["label"].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES].index

removed_classes = class_counts[class_counts < MIN_SAMPLES]

if len(removed_classes) > 0:
    print("\n⚠️ Removing rare classes (too few samples):")
    print(removed_classes)

df = df[df["label"].isin(valid_classes)]

print("\n✅ Class distribution after filtering:")
print(df["label"].value_counts())



# TRAIN / TEST SPLIT (SAFE)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)



# PIPELINE (TF-IDF + LOGISTIC)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(
        max_iter=3000,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1
    ))
])



# CROSS-VALIDATION

cv_scores = cross_val_score(
    pipeline,
    df["text"],
    df["label"],
    cv=5,
    scoring="f1_macro"
)

print("\n📈 Cross-Validation F1 Scores:", cv_scores)
print("📈 Mean F1 Score:", cv_scores.mean())



# TRAIN MODEL

pipeline.fit(X_train, y_train)



# EVALUATION

y_pred = pipeline.predict(X_test)

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

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# SAVE PIPELINE

joblib.dump(pipeline, "document_classifier_pipeline.pkl")
print("\n✅ Model saved as document_classifier_pipeline.pkl")
