import nltk
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK tokenizer verisini sessizce indir
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


@st.cache_resource(show_spinner=False)
def load_extractive_model():
    """Sentence-BERT modelini cache'le ve yükle (CPU)."""
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def extractive_summarize(text: str, top_n: int = 3) -> str:
    """
    Extractive özetleme:
    - Cümle embeddingleri üret
    - Centroid ile cosine similarity hesapla
    - En yüksek skorlu top_n cümleyi orijinal sırayla döndür
    """
    if not text or not text.strip():
        return ""

    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return ""
    if len(sentences) <= top_n:
        return " ".join(sentences)

    model = load_extractive_model()
    embeddings = model.encode(sentences, convert_to_numpy=True)

    # Centroid (tüm cümlelerin ortalama vektörü) ile benzerlik
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    similarities = cosine_similarity(centroid, embeddings)[0]

    # En yüksek skorlu top_n index'i al ve orijinal sıraya göre diz
    top_indices = np.argsort(similarities)[-top_n:]
    top_indices = sorted(top_indices)

    selected = [sentences[i] for i in top_indices]
    return " ".join(selected)