import streamlit as st
from PyPDF2 import PdfReader
from utils.summarizer import load_summarizer, run_abstractive, run_extractive
from utils.ner import load_ner, run_ner
from utils.visualization import generate_wordcloud
from rouge_score import rouge_scorer

st.set_page_config(page_title="Türkçe NLP Pro", layout="wide")

st.title("🚀 Türkçe NLP Analiz Sistemi")

# Sidebar
st.sidebar.title("⚙️ Kontrol Paneli")
input_method = st.sidebar.radio("Veri Kaynağı", ["Metin Gir", "PDF Yükle"])

raw_text = ""

if input_method == "PDF Yükle":
    file = st.sidebar.file_uploader("PDF Yükle", type="pdf")
    if file:
        reader = PdfReader(file)
        for page in reader.pages:
            raw_text += page.extract_text() + " "
else:
    raw_text = st.sidebar.text_area("Metni gir:", height=300)

if len(raw_text) > 100:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Özetleme",
        "🔍 NER",
        "📊 Görselleştirme",
        "📏 ROUGE"
    ])

    # ÖZET
    with tab1:
        if st.button("Özetle"):
            tokenizer, model = load_summarizer()

            abs_sum = run_abstractive(raw_text, tokenizer, model)
            ext_sum = run_extractive(raw_text)

            col1, col2 = st.columns(2)
            col1.write("### 🤖 Yapay Zeka")
            col1.write(abs_sum)

            col2.write("### ✂️ Extractive")
            col2.write(ext_sum)

    # NER
    with tab2:
        if st.button("Varlıkları Bul"):
            ner_model = load_ner()
            ents = run_ner(raw_text, ner_model)

            st.write("### Kişiler:", ", ".join(ents["Kişi"]))
            st.write("### Kurumlar:", ", ".join(ents["Kurum"]))
            st.write("### Yerler:", ", ".join(ents["Yer"]))

    # WORDCLOUD
    with tab3:
        if st.button("Kelime Bulutu"):
            fig = generate_wordcloud(raw_text)
            st.pyplot(fig)

    # ROUGE
    with tab4:
        if st.button("ROUGE Hesapla"):
            tokenizer, model = load_summarizer()

            ref = run_extractive(raw_text)
            hyp = run_abstractive(raw_text, tokenizer, model)

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
            scores = scorer.score(ref, hyp)

            st.write(scores)

else:
    st.warning("En az 100 karakter gir.")