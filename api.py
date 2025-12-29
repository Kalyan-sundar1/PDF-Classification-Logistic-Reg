from flask import Flask, request, jsonify
import pdfplumber
import joblib


# -----------------------------
# Load trained model & vectorizer
# -----------------------------
model = joblib.load("document_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# -----------------------------
# Label mapping
# -----------------------------
label_map = {
    0: "Resume",
    1: "Invoice",
    2: "Article",
    3: "Annual Report",
    4: "Contract",
    5: "Receipt"
}


# -----------------------------
# PDF text extraction
# -----------------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# -----------------------------
# Flask app initialization
# -----------------------------
app = Flask(__name__)


# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]

    text = extract_text(pdf_file)
    if not text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    return jsonify({
        "document_type": label_map[prediction],
        "confidence": round(float(confidence), 2)
    })


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
