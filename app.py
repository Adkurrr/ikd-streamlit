import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model dan tokenizer dari Hugging Face Hub
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_sentiment_analysis")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_sentiment_analysis")
    return model, tokenizer

model, tokenizer = load_model()

# Judul aplikasi
st.title("Analisis Sentimen Ulasan IKD 🇮🇩")
st.write("Masukkan ulasan aplikasi Identitas Kependudukan Digital untuk dianalisis sentimennya.")

# Input dari pengguna
text_input = st.text_area("Tulis ulasan di sini:")

# Tombol prediksi
if st.button("Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        # Tokenisasi
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        
        # Prediksi
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        # Label sentimen (2 kelas)
        label_map = {
            0: "Negatif",
            1: "Positif"
        }

        sentiment = label_map.get(pred, "Tidak diketahui")
        
        st.subheader("Hasil Analisis")
        st.write(f"**Sentimen:** {sentiment}")
        st.write(f"**Kepercayaan:** {confidence * 100:.2f}%")
