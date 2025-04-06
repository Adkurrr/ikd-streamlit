import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from PIL import Image
import os
import gdown

# === Download model.safetensors dari Google Drive jika belum ada ===
@st.cache_resource
def download_model_file():
    model_path = "saved_model/model.safetensors"
    if not os.path.exists(model_path):
        st.info("Mengunduh model dari Google Drive...")
        gdown.download(
            id="11O8IkrorN_fAFiqXKK3ARdqi0GfOJDy5",
            output=model_path,
            quiet=False
        )
    return model_path

# === Load model dan tokenizer ===
@st.cache_resource
def load_model():
    download_model_file()  # pastikan file ada dulu
    model = BertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    return model, tokenizer

model, tokenizer = load_model()

# Sidebar navigasi
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih halaman:", ["Tentang Aplikasi", "Dataset & Hasil", "Cek Sentimen"])

# --- Halaman 1: Tentang Aplikasi ---
if halaman == "Tentang Aplikasi":
    st.title("Aplikasi Identitas Kependudukan Digital")
    
    # Gambar aplikasi (ganti dengan path file kamu)
    if os.path.exists("gambar_aplikasi.png"):
        image = Image.open("gambar_aplikasi.png")
        st.image(image, use_column_width=True)
    
    st.markdown("""
    Aplikasi **Identitas Kependudukan Digital** adalah aplikasi resmi dari pemerintah Indonesia
    yang memungkinkan warga untuk menyimpan KTP dalam bentuk digital, mempermudah akses ke layanan publik,
    serta mempercepat proses administrasi.

    Dalam proyek ini, kita menganalisis ulasan pengguna terhadap aplikasi tersebut untuk memahami kepuasan masyarakat.
    """)

# --- Halaman 2: Dataset dan Hasil ---
elif halaman == "Dataset & Hasil":
    st.title("Dataset dan Hasil Analisis")
    
    st.markdown("""
    Dataset yang digunakan merupakan kumpulan ulasan dari Google Play Store terhadap aplikasi Identitas Kependudukan Digital.
    Dataset ini telah melalui tahap preprocessing sebelum dilakukan analisis sentimen menggunakan model BERT.
    """)
    
    st.header("Visualisasi Dataset dan Model")
    
    # Visualisasi 1
    st.subheader("Distribusi Sentimen")
    if os.path.exists("visualisasi_sentimen.png"):
        st.image("visualisasi_sentimen.png", caption="Distribusi sentimen positif dan negatif")
    
    # Visualisasi 2
    st.subheader("Akurasi Model")
    if os.path.exists("akurasi_model.png"):
        st.image("akurasi_model.png", caption="Perbandingan akurasi model")

# --- Halaman 3: Cek Sentimen ---
elif halaman == "Cek Sentimen":
    st.title("Cek Sentimen Ulasan")

    text = st.text_area("Masukkan ulasan pengguna:")

    if st.button("Prediksi Sentimen"):
        if text.strip() == "":
            st.warning("Silakan masukkan ulasan terlebih dahulu.")
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()

            label = "Positif" if pred == 1 else "Negatif"
            if label == "Positif":
                st.success(f"Sentimen: {label}")
            else:
                st.error(f"Sentimen: {label}")
