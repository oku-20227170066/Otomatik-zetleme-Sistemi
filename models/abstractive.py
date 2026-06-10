import os
import nltk
import streamlit as st
from transformers import pipeline

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

TRAINED_MODEL_PATH = "./trained_model"
BASE_MODEL = "google/mt5-base"


def _model_exists(path: str) -> bool:
    """Klasörde model dosyası var mı kontrol et."""
    if not os.path.isdir(path):
        return False
    has_weights = (
        os.path.exists(os.path.join(path, "pytorch_model.bin"))
        or os.path.exists(os.path.join(path, "model.safetensors"))
    )
    has_config = os.path.exists(os.path.join(path, "config.json"))
    return has_weights and has_config


@st.cache_resource(show_spinner=False)
def load_abstractive_model():
    """
    Eğitilmiş model varsa onu yükle, yoksa base mT5'i kullan.
    """
    if _model_exists(TRAINED_MODEL_PATH):
        model_path = TRAINED_MODEL_PATH
        print(f"🎓 Eğitilmiş model yüklendi: {model_path}")
    else:
        model_path = BASE_MODEL
        print(f"📦 Base model yüklendi: {model_path}")

    return pipeline(
    "text2text-generation",
    model=model_path,
    tokenizer=model_path,
    device=-1,
    torch_dtype="auto",
    use_fast=False,
    )



def abstractive_summarize(
    text: str,
    max_chunk_sentences: int = 5,
    max_new_tokens: int = 128,
) -> str:
    if not text or not text.strip():
        return ""

    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return ""

    chunks = []
    for i in range(0, len(sentences), max_chunk_sentences):
        chunk = " ".join(sentences[i : i + max_chunk_sentences])
        chunks.append(chunk)

    model = load_abstractive_model()
    summaries = []

    for chunk in chunks:
        prompt = f"summarize: {chunk}"
        try:
            output = model(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
            )
            summaries.append(output[0]["generated_text"].strip())
        except Exception as e:
            summaries.append(f"[Chunk hatası: {str(e)}]")

    return " ".join(summaries)