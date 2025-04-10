import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re
import spacy
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#Download fungsi yg dibutuhkan
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import spacy.cli
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
# Inisialisasi resource
@st.cache_resource
def load_resources():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_sentiment_analysis")
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_sentiment_analysis")

    stop_words = set(stopwords.words('indonesian'))
    custom_stopwords = {'nya', 'yg', 'kali', 'bgt', 'mls'}
    stop_words.update(custom_stopwords)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return nlp, tokenizer, model, stop_words, stemmer

nlp, tokenizer, model, stop_words, stemmer = load_resources()

# --- Preprocessing sesuai training ---
def cleansing_text(review):
    if not isinstance(review, str):
        return ""
    review = review.lower()
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    return review

def tokenize_text(text):
    return [token.text for token in nlp(text)]

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stemming_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def preprocess_input(text):
    clean = cleansing_text(text)
    tokens = tokenize_text(clean)
    tokens_no_stopword = remove_stopwords(tokens)
    stemmed_tokens = stemming_tokens(tokens_no_stopword)
    return " ".join(stemmed_tokens)

# --- Streamlit UI ---
st.title("Analisis Sentimen Ulasan IKD 🇮🇩")
st.write("Masukkan ulasan aplikasi Identitas Kependudukan Digital untuk dianalisis sentimennya.")

text_input = st.text_area("Tulis ulasan di sini:")

if st.button("Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        # Preprocess dulu
        preprocessed_text = preprocess_input(text_input)

        # Tokenisasi dan prediksi
        inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        # Mapping label: pastikan sesuai config.json
        label_map = {
            0: "Negatif",
            1: "Positif"
        }

        sentiment = label_map.get(pred, "Tidak diketahui")
        
        st.subheader("Hasil Analisis")
        st.write(f"**Sentimen:** {sentiment}")
        st.write(f"**Kepercayaan:** {confidence * 100:.2f}%")
