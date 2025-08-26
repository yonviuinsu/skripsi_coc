import os
import joblib
import numpy as np
from flask import Flask, render_template, request

# Inisialisasi Flask
app = Flask(__name__)

# === Load artefak model ===
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# Load semua model dasar dan meta model
models = {
    "SVM": joblib.load("models/SVM.joblib"),
    "Naive Bayes": joblib.load("models/NaiveBayes.joblib"),
    "Random Forest": joblib.load("models/RandomForest.joblib"),
    "Stacking": joblib.load("models/stack_model.joblib")
}

# === Fungsi Preprocessing ===
def preprocess_text(text: str) -> str:
    return text.lower().strip()

# === Routing utama ===
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_model = None
    comment = None

    if request.method == "POST":
        comment = request.form["comment"]
        selected_model = request.form["model"]

        # Preprocessing teks input
        clean_comment = preprocess_text(comment)
        vectorized_comment = tfidf.transform([clean_comment])

        # Jika user memilih stacking, hitung probabilitas dari semua model dasar
        if selected_model == "Stacking":
            proba_list = []
            for name in ["SVM", "Naive Bayes", "Random Forest"]:
                model = models[name]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vectorized_comment)
                else:
                    # Fallback jika model tidak punya predict_proba
                    decision = model.decision_function(vectorized_comment)
                    proba = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
                proba_list.append(proba)

            stack_features = np.hstack(proba_list)
            pred_encoded = models["Stacking"].predict(stack_features)
        else:
            model = models[selected_model]
            pred_encoded = model.predict(vectorized_comment)

        # Decode label hasil prediksi
        prediction = label_encoder.inverse_transform(pred_encoded)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        selected_model=selected_model,
        comment=comment
    )

# === Jalankan Flask ===
if __name__ == "__main__":
    app.run(debug=True)
