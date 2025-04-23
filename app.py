import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from huggingface_hub import hf_hub_download

# === Inisialisasi stopword remover dan stemmer
stop_factory = StopWordRemoverFactory()
stop_remover = stop_factory.create_stop_word_remover()
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# === Praproses sesuai training
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)  # hapus URL
    text = re.sub(r'@\w+|#\w+', '', text)  # hapus mention & hashtag
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # hapus tanda baca
    text = re.sub(r'\d+', '', text)  # hapus angka
    text = text.strip()
    text = stop_remover.remove(text)
    text = stemmer.stem(text)
    return text

# ====== Load Models from Hugging Face Hub ======
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

# ====== Predict Functions ======
def predict_with_bert(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
    return pred, probs.numpy()

def predict_with_model(text, model):
    preprocessed = preprocess(text)
    return model.predict([preprocessed])[0]

# ====== UI Streamlit ======
st.title("Aplikasi Analisis Sentimen IKD 🇮🇩")
st.write("Masukkan ulasan, pilih model, dan lihat prediksi sentimennya!")

text_input = st.text_area("Masukkan ulasan:", "")
model_choice = st.selectbox("Pilih Model", [
    "BERT Finetuned", "BERT Pretrained", "Logistic Regression", "SVM"
])

if st.button("🔍 Prediksi Sentimen"):
    if not text_input.strip():
        st.warning("Silakan isi ulasan terlebih dahulu.")
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

        sentimen_label = "Positif" if str(label) in ["1", "positif", "positive"] else "Negatif"
        st.success(f"Prediksi Sentimen: {sentimen_label}")
