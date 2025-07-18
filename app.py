import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from huggingface_hub import hf_hub_download
from PIL import Image
import requests
from io import BytesIO
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk


# === Setup Stopwords dan Stemmer ===
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

#===============================================================================================================================================
# === Fungsi Praproses Teks ===
def preprocess_full(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

def preprocess_woss(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text

def preprocess_sro(text):
    text = text.lower()  
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) 
    tokens = text.split()  
    cleaned = [word for word in tokens if word not in stop_words]  
    return ' '.join(cleaned)

def preprocess_so(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    cleaned = [stemmer.stem(word) for word in tokens]
    return ' '.join(cleaned)
 #===========================================================================================================================================   
# === Load Models ===
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))
@st.cache_resource
def load_bert_finetuned_s1():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_ft_fullpraproses")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_ft_fullpraproses")
    return model, tokenizer
@st.cache_resource
def load_bert_pretrained_s1():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_pretrained_fullpraproses")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_pretrained_fullpraproses")
    return model, tokenizer
@st.cache_resource
def load_lr_model_s1():
    file_path = hf_hub_download(repo_id="Adkurrr/lr-SVM-fullpraproses", filename="lr_model.pkl")
    return joblib.load(file_path)
@st.cache_resource
def load_svm_model_s1():
    file_path = hf_hub_download(repo_id="Adkurrr/Lr-SVM-fullpraproses", filename="svm_model.pkl")
    return joblib.load(file_path)



@st.cache_resource
def load_bert_finetuned_s2():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_ft_woss")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_ft_woss")
    return model, tokenizer
@st.cache_resource
def load_bert_pretrained_s2():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_pretrained_woss")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_pretrained_woss")
    return model, tokenizer
@st.cache_resource
def load_lr_model_s2():
    file_path = hf_hub_download(repo_id="Adkurrr/lr-SVM-woss", filename="lr_model_woss.pkl")
    return joblib.load(file_path)
@st.cache_resource
def load_svm_model_s2():
    file_path = hf_hub_download(repo_id="Adkurrr/Lr-SVM-woss", filename="svm_model_woss.pkl")
    return joblib.load(file_path)



@st.cache_resource
def load_bert_finetuned_s3():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_ft_fullpraproses")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_ft_StopwordRemovalOnly")
    return model, tokenizer
@st.cache_resource
def load_bert_pretrained_s3():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_pretrained_StopwordRemovalOnly")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_pretrained_StopwordRemovalOnly")
    return model, tokenizer
@st.cache_resource
def load_lr_model_s3():
    file_path = hf_hub_download(repo_id="Adkurrr/lr-SVM-StopwordRemovalOnly", filename="lr_model.pkl")
    return joblib.load(file_path)
@st.cache_resource
def load_svm_model_s3():
    file_path = hf_hub_download(repo_id="Adkurrr/Lr-SVM-StopwordRemovalOnly", filename="svm_model.pkl")
    return joblib.load(file_path)

@st.cache_resource
def load_bert_finetuned_s4():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_ft_StemmingOnly")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_ft_StemmingOnly")
    return model, tokenizer
@st.cache_resource
def load_bert_pretrained_s4():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_pretrained_StemmingOnly")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_pretrained_StemmingOnly")
    return model, tokenizer

@st.cache_resource
def load_lr_model_s4():
    file_path = hf_hub_download(repo_id="Adkurrr/lr-SVM-StemmingOnly", filename="lr_model.pkl")
    return joblib.load(file_path)

@st.cache_resource
def load_svm_model_s4():
    file_path = hf_hub_download(repo_id="Adkurrr/Lr-SVM-StemmingOnly", filename="svm_model.pkl")
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

menu = st.sidebar.radio("Skenario", ["Stopword Removal dan Stemming", "Tanpa Stopword Removal dan Stemming", "Stopword Removal", "Stemming"])

if menu == "Stopword Removal dan Stemming":
    st.title("Prediksi Sentimen Ulasan IKD")
    st.write("Skenario : Praproses lengkap (Cleansing, Tokenisasi, Stopwords Removal dan Stemming)")

    text_input = st.text_area("Masukkan ulasan:", "")
    model_choice = st.selectbox("Pilih Model", [
        "BERT Finetuned", "BERT Pretrained", "Logistic Regression", "Support Vector Machine"])

    if st.button("🔍 Prediksi Sentimen"):
        if not text_input.strip():
            st.warning("⚠️ Ulasan Tidak Boleh Kosong")
        else:
            # Praproses semua input sebelum diprediksi
            processed_text = preprocess_full(text_input)

            if model_choice == "BERT Finetuned":
                model, tokenizer = load_bert_finetuned_s1()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "BERT Pretrained":
                model, tokenizer = load_bert_pretrained_s1()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "Logistic Regression":
                model = load_lr_model_s1()
                label = predict_with_model(processed_text, model)
            elif model_choice == "Support Vector Machine":
                model = load_svm_model_s1()
                label = predict_with_model(processed_text, model)
            else:
                label = "?"

            sentimen_label = "Positif" if str(label) in ["1", "positif", "positive"] else "Negatif"
            st.success(f"Prediksi Sentimen: {sentimen_label}")

elif menu == "Tanpa Stopword Removal dan Stemming":
    st.title("Prediksi Sentimen Ulasan IKD")
    st.write("Skenario : Tanpa Stopword Removal dan Stemming")

    text_input = st.text_area("Masukkan ulasan:", "")
    model_choice = st.selectbox("Pilih Model", [
        "BERT Finetuned", "BERT Pretrained", "Logistic Regression", "Support Vector Machine"])

    if st.button("🔍 Prediksi Sentimen"):
        if not text_input.strip():
            st.warning("⚠️ Ulasan Tidak Boleh Kosong")
        else:
            # Praproses semua input sebelum diprediksi
            processed_text = preprocess_woss(text_input)

            if model_choice == "BERT Finetuned":
                model, tokenizer = load_bert_finetuned_s2()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "BERT Pretrained":
                model, tokenizer = load_bert_pretrained_s2()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "Logistic Regression":
                model = load_lr_model_s2()
                label = predict_with_model(processed_text, model)
            elif model_choice == "Support Vector Machine":
                model = load_svm_model_s2()
                label = predict_with_model(processed_text, model)
            else:
                label = "?"

            sentimen_label = "Positif" if str(label) in ["1", "positif", "positive"] else "Negatif"
            st.success(f"Prediksi Sentimen: {sentimen_label}")
    
elif menu == "Stopword Removal":
    st.title("Prediksi Sentimen Ulasan IKD")
    st.write("Skenario : Menghapus Stopword Removal")
    text_input = st.text_area("Masukkan ulasan:", "")
    model_choice = st.selectbox("Pilih Model", [
    "BERT Finetuned", "BERT Pretrained", "Logistic Regression", "Support Vector Machine"])

    if st.button("🔍 Prediksi Sentimen"):
        if not text_input.strip():
            st.warning("⚠️ Ulasan Tidak Boleh Kosong")
        else:
            # Praproses semua input sebelum diprediksi
            processed_text = preprocess_sro(text_input)
    
            if model_choice == "BERT Finetuned":
                model, tokenizer = load_bert_finetuned_s3()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "BERT Pretrained":
                model, tokenizer = load_bert_pretrained_s3()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "Logistic Regression":
                model = load_lr_model_s3()
                label = predict_with_model(processed_text, model)
            elif model_choice == "SVM":
                model = load_svm_model_s3()
                label = predict_with_model(processed_text, model)
            else:
                label = "?"
    
            sentimen_label = "Positif" if str(label) in ["1", "positif", "positive"] else "Negatif"
            st.success(f"Prediksi Sentimen: {sentimen_label}")
elif menu == "Stemming":
    st.title("Prediksi Sentimen Ulasan IKD")
    st.write("Skenario : Stemming")
    text_input = st.text_area("Masukkan ulasan:", "")
    model_choice = st.selectbox("Pilih Model", [
    "BERT Finetuned", "BERT Pretrained", "Logistic Regression", "Support Vector Machine"])

    if st.button("🔍 Prediksi Sentimen"):
        if not text_input.strip():
            st.warning("⚠️ Ulasan Tidak Boleh Kosong")
        else:
            # Praproses semua input sebelum diprediksi
            processed_text = preprocess_so(text_input)
    
            if model_choice == "BERT Finetuned":
                model, tokenizer = load_bert_finetuned_s4()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "BERT Pretrained":
                model, tokenizer = load_bert_pretrained_s4()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "Logistic Regression":
                model = load_lr_model_s4()
                label = predict_with_model(processed_text, model)
            elif model_choice == "Support Vector Machine":
                model = load_svm_model_s4()
                label = predict_with_model(processed_text, model)
            else:
                label = "?"
    
            sentimen_label = "Positif" if str(label) in ["1", "positif", "positive"] else "Negatif"
            st.success(f"Prediksi Sentimen: {sentimen_label}")
