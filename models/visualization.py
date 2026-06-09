import string

import nltk
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


def generate_wordcloud(text: str):
    """
    Metinden Türkçe stopwords'leri atıp WordCloud görseli üret.
    """
    if not text or not text.strip():
        return None

    try:
        from nltk.corpus import stopwords

        turkish_stopwords = set(stopwords.words("turkish"))
        # Noktalama işaretlerini de stopwords'e ekle
        turkish_stopwords.update(string.punctuation)

        # Tokenize ve filtrele
        words = nltk.word_tokenize(text.lower())
        filtered = [
            w
            for w in words
            if w.isalpha() and w not in turkish_stopwords and len(w) > 2
        ]

        if not filtered:
            return None

        clean_text = " ".join(filtered)

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=turkish_stopwords,
            max_words=100,
            contour_width=1,
            colormap="viridis",
        )
        wc.generate(clean_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout()
        return fig

    except Exception as e:
        raise RuntimeError(f"WordCloud oluşturma hatası: {str(e)}")