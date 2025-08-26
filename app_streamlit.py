import streamlit as st
import joblib
import numpy as np

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

# === Konfigurasi Halaman Streamlit ===
st.set_page_config(page_title="Aplikasi Analisis Sentimen", layout="wide")
st.title("📊 Aplikasi Analisis Sentimen")
st.markdown("Pilih model dan masukkan komentar untuk memprediksi sentimen.")

# === Form Input ===
with st.form("sentiment_form"):
    comment = st.text_area("Masukkan komentar:", height=150)
    selected_model = st.selectbox(
        "Pilih model yang digunakan:",
        ["SVM", "Naive Bayes", "Random Forest", "Stacking"]
    )
    submitted = st.form_submit_button("Prediksi")

# === Proses Prediksi ===
if submitted:
    if not comment.strip():
        st.warning("⚠️ Teks komentar tidak boleh kosong.")
    else:
        clean_comment = preprocess_text(comment)
        vectorized_comment = tfidf.transform([clean_comment])

        # Jika user memilih stacking
        if selected_model == "Stacking":
            proba_list = []
            for name in ["SVM", "Naive Bayes", "Random Forest"]:
                model = models[name]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vectorized_comment)
                else:
                    decision = model.decision_function(vectorized_comment)
                    proba = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
                proba_list.append(proba)

            stack_features = np.hstack(proba_list)
            pred_encoded = models["Stacking"].predict(stack_features)
        else:
            model = models[selected_model]
            pred_encoded = model.predict(vectorized_comment)

        prediction = label_encoder.inverse_transform(pred_encoded)[0]

        # === Hasil Prediksi ===
        st.success(f"**Hasil Prediksi Sentimen:** {prediction}")

        # Menampilkan informasi tambahan jika model bukan stacking
        if selected_model != "Stacking" and hasattr(models[selected_model], "predict_proba"):
            probs = models[selected_model].predict_proba(vectorized_comment)[0]
            classes = label_encoder.classes_
            st.subheader("📌 Probabilitas Prediksi")
            st.dataframe({
                "Sentimen": classes,
                "Probabilitas": np.round(probs, 3)
            })
