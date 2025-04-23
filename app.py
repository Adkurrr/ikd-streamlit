import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from huggingface_hub import hf_hub_download
from PIL import Image
import request
from io import BytesIO

# === Load Models ===
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


@st.cache_resource
def load_bert_finetuned():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_ft_fullpreprocessing")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_ft_fullpreprocessing")
    return model, tokenizer

@st.cache_resource
def load_bert_pretrained():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_pretrained")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_pretrained")
    return model, tokenizer

@st.cache_resource
def load_lr_model():
    file_path = hf_hub_download(repo_id="Adkurrr/LogisticRegression_and_SVM", filename="lr_model.pkl")
    return joblib.load(file_path)

@st.cache_resource
def load_svm_model():
    file_path = hf_hub_download(repo_id="Adkurrr/LogisticRegression_and_SVM", filename="svm_model.pkl")
    return joblib.load(file_path)

# === Prediction Functions ===
def predict_with_bert(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
    return pred, probs.numpy()

def predict_with_model(text, model):
    return model.predict([text])[0]

# === Halaman Utama Streamlit ===
st.set_page_config(page_title="Analisis Sentimen Identitas Kependudukan Digital", layout="wide")

menu = st.sidebar.radio("Navigasi", ["Eksplorasi Data", "Prediksi Sentimen"])

if menu == "Eksplorasi Data":
    st.title("Eksplorasi Dataset Ulasan Identitas Kependudukan Digital (IKD)")

    st.markdown("""
    Dataset yang digunakan merupakan ulasan pengguna aplikasi Identitas Kependudukan Digital (IKD) dari Google Play Store.
    Data telah diproses melalui tahapan pembersihan, tokenisasi, stopwords removal, dan stemming.
    """)

    # Load dataset lokal atau Hugging Face jika perlu
    df = pd.read_csv("https://github.com/Adkurrr/digital-residence-identity-sentiment-analysis/blob/main/Dataset/dataset%20final.xlsx?raw=True")  # Ganti dengan path dataset kamu

    st.subheader("Contoh Data")
    st.dataframe(df.sample(5))

    st.subheader("Distribusi Sentimen dalam Dataset")
    img1 = load_image_from_url("https://raw.githubusercontent.com/Adkurrr/ikd-streamlit/main/distribusi%20data.png")
    st.image(img1, caption="Distribusi Data Sentimen")

    st.subheader("Wordcloud Kata-Kata Umum")
    img2 = load_image_from_url("https://raw.githubusercontent.com/Adkurrr/ikd-streamlit/main/wordcloud.png")
    st.image(img2, caption="Wordcloud Ulasan")

    st.subheader("Perbandingan Model")
    comparison_data = {
        'Model': ["BERT Finetuned", "BERT Pretrained", "Logistic Regression", "SVM"],
        'Akurasi': [0.89, 0.82, 0.79, 0.80],
        'F1-Score': [0.89, 0.81, 0.78, 0.79]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

elif menu == "🤖 Prediksi Sentimen":
    st.title("Prediksi Sentimen Ulasan IKD 🇮🇩")
    st.write("Masukkan ulasan, pilih model, dan lihat hasil prediksinya!")

    text_input = st.text_area("📝 Masukkan ulasan:", "")
    model_choice = st.selectbox("🧠 Pilih Model", [
        "BERT Finetuned", "BERT Pretrained", "Logistic Regression", "SVM"
    ])

    if st.button("🔍 Prediksi Sentimen"):
        if not text_input.strip():
            st.warning("⚠️ Silakan isi ulasan terlebih dahulu.")
        else:
            if model_choice == "BERT Finetuned":
                model, tokenizer = load_bert_finetuned()
                label, probs = predict_with_bert(text_input, model, tokenizer)
            elif model_choice == "BERT Pretrained":
                model, tokenizer = load_bert_pretrained()
                label, probs = predict_with_bert(text_input, model, tokenizer)
            elif model_choice == "Logistic Regression":
                model = load_lr_model()
                label = predict_with_model(text_input, model)
            elif model_choice == "SVM":
                model = load_svm_model()
                label = predict_with_model(text_input, model)
            else:
                label = "?"

            sentimen_label = "Positif 😄" if str(label) in ["1", "positif", "positive"] else "Negatif 😠"
            st.success(f"✅ Prediksi Sentimen: {sentimen_label}")
