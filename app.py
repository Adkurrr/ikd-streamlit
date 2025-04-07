import streamlit as st
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Caching model dan tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("Adkurrr/ikd_streamlit")
    model = BertForSequenceClassification.from_pretrained("Adkurrr/ikd_streamlit")
    return tokenizer, model

# Inisialisasi model dan tokenizer
tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# UI
st.title("Analisis Sentimen Ulasan Aplikasi IKD")
st.write("Masukkan ulasan pengguna untuk melihat sentimennya (Positif atau Negatif)")

user_input = st.text_area("Masukkan ulasan:")
button = st.button("Analisa")

# Label sentimen
sentiment_labels = {
    1: 'Positif',
    0: 'Negatif'
}

# Prediksi
if user_input and button:
    with st.spinner("Menganalisis..."):
        # Tokenisasi dan konversi ke tensor
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Inferensi
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        
        st.success(f"Prediksi Sentimen: **{sentiment_labels[prediction]}**")
        st.write("Logits:", logits.cpu().numpy())
