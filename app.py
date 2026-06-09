import io
import os
import subprocess
import sys
import tempfile

import pandas as pd
import streamlit as st

from main import summarize_text

st.set_page_config(page_title="NLP Özetleme Sistemi", layout="wide")

st.title("📝 NLP Otomatik Özetleme Sistemi")
st.markdown(
    "**Extractive** + **Abstractive** + **ROUGE** + **NER** + **WordCloud**"
)

with st.sidebar:
    st.header("⚙️ Ayarlar")
    top_n = st.slider("Extractive cümle sayısı", min_value=1, max_value=10, value=3)

    if os.path.isdir("./trained_model") and os.path.exists(
        os.path.join("./trained_model", "config.json")
    ):
        st.success("🎓 Eğitilmiş model aktif!")

    st.info(
        "İlk çalıştırmada modeller otomatik indirilecektir. "
        "Bu işlem birkaç dakika sürebilir."
    )


# ============================================================
# FONKSİYON TANIMLARI (ÖNCE)
# ============================================================
def _show_results(results: dict):
    """Tek metin sonuçlarını göster."""
    st.success("✅ Analiz tamamlandı!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Extractive Özet")
        st.info(results["extractive"])
    with col2:
        st.subheader("✨ Abstractive Özet")
        st.success(results["abstractive"])

    st.subheader("📊 ROUGE Skorları (Abstractive vs Extractive)")
    r = results["rouge"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ROUGE-1", f"{r['rouge1']:.4f}")
    c2.metric("ROUGE-2", f"{r['rouge2']:.4f}")
    c3.metric("ROUGE-L", f"{r['rougeL']:.4f}")

    st.subheader("🔍 Named Entity Recognition (NER)")
    if results["entities"]:
        st.dataframe(results["entities"], use_container_width=True)

        per = list(
            dict.fromkeys([e["word"] for e in results["entities"] if e["type"] == "PER"])
        )
        org = list(
            dict.fromkeys([e["word"] for e in results["entities"] if e["type"] == "ORG"])
        )
        loc = list(
            dict.fromkeys([e["word"] for e in results["entities"] if e["type"] == "LOC"])
        )

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.markdown("**👤 Kişiler**")
            for w in per:
                st.markdown(
                    f"<span style='background:#dbeafe;color:#000000;padding:4px 8px;border-radius:12px;margin:2px;display:inline-block'>{w}</span>",
                    unsafe_allow_html=True,
                )
        with bc2:
            st.markdown("**🏢 Kurumlar**")
            for w in org:
                st.markdown(
                    f"<span style='background:#dcfce7;color:#000000;padding:4px 8px;border-radius:12px;margin:2px;display:inline-block'>{w}</span>",
                    unsafe_allow_html=True,
                )
        with bc3:
            st.markdown("**📍 Yerler**")
            for w in loc:
                st.markdown(
                    f"<span style='background:#ffedd5;color:#000000;padding:4px 8px;border-radius:12px;margin:2px;display:inline-block'>{w}</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.warning("Entity bulunamadı.")

    st.subheader("📊 Kelime Frekansı (WordCloud)")
    if results["wordcloud"]:
        st.pyplot(results["wordcloud"])
    else:
        st.warning("WordCloud oluşturulamadı.")


# ============================================================
# SEKME SİSTEMİ (SONRA)
# ============================================================
tab_single, tab_batch, tab_train = st.tabs(
    ["📝 Tek Metin", "📁 Toplu İşlem", "🎓 Model Eğitimi"]
)

with tab_single:
    text_input = st.text_area(
        "Özetlenecek metni girin:",
        height=280,
        placeholder="Buraya analiz edilecek metni yapıştırın...",
    )

    if st.button("🚀 Özetle ve Analiz Et", type="primary", use_container_width=True):
        if not text_input or not text_input.strip():
            st.error("❌ Lütfen özetlenecek bir metin girin.")
        else:
            with st.spinner("Analiz yapılıyor..."):
                try:
                    results = summarize_text(text_input, top_n=top_n)
                    _show_results(results)
                except Exception as e:
                    st.error(f"❌ Hata: {str(e)}")


with tab_batch:
    st.header("📁 Toplu Özetleme")
    st.markdown("**Beklenen format:** `text` sütunu içeren CSV veya JSON.")

    uploaded_file = st.file_uploader("Dosya yükleyin", type=["csv", "json"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            if "text" not in df.columns:
                st.error("❌ 'text' sütunu bulunamadı!")
                st.stop()

            total = len(df)
            st.info(f"📄 {total} satır bulundu.")

            if st.button("▶️ Toplu Analizi Başlat", type="primary"):
                results_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, row in df.iterrows():
                    raw_text = str(row["text"]) if pd.notna(row["text"]) else ""
                    status_text.text(f"İşleniyor: {idx + 1}/{total}")

                    if not raw_text.strip():
                        results_list.append(
                            {
                                "original": raw_text,
                                "extractive": "",
                                "abstractive": "",
                                "rouge1": 0.0,
                                "rouge2": 0.0,
                                "rougeL": 0.0,
                                "entities": "",
                                "error": "Boş metin",
                            }
                        )
                    else:
                        try:
                            res = summarize_text(raw_text, top_n=top_n)
                            ent_str = ", ".join(
                                [f"{e['word']}({e['type']})" for e in res["entities"]]
                            )
                            results_list.append(
                                {
                                    "original": raw_text,
                                    "extractive": res["extractive"],
                                    "abstractive": res["abstractive"],
                                    "rouge1": res["rouge"]["rouge1"],
                                    "rouge2": res["rouge"]["rouge2"],
                                    "rougeL": res["rouge"]["rougeL"],
                                    "entities": ent_str,
                                    "error": "",
                                }
                            )
                        except Exception as e:
                            results_list.append(
                                {
                                    "original": raw_text,
                                    "extractive": "",
                                    "abstractive": "",
                                    "rouge1": 0.0,
                                    "rouge2": 0.0,
                                    "rougeL": 0.0,
                                    "entities": "",
                                    "error": str(e),
                                }
                            )

                    progress_bar.progress((idx + 1) / total)

                status_text.empty()
                progress_bar.empty()

                result_df = pd.DataFrame(results_list)
                st.success(f"✅ {total} satır işlendi!")
                st.dataframe(result_df, use_container_width=True)

                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                st.download_button(
                    label="⬇️ Sonuçları CSV İndir",
                    data=csv_buffer.getvalue().encode("utf-8-sig"),
                    file_name="ozetleme_sonuclari.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"❌ Dosya okuma hatası: {str(e)}")


with tab_train:
    st.header("🎓 Abstractive Model Eğitimi (Fine-tuning)")
    st.markdown(
        """
        **Veriseti formatı:** `text` ve `summary` sütunları.
        
        ⚠️ **CPU'da fine-tuning çok yavaştır.** 1000 örnek ≈ 3-5 saat sürebilir.
        """
    )

    train_file = st.file_uploader(
        "Eğitim veriseti (CSV/JSON)", type=["csv", "json"]
    )

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epoch", min_value=1, max_value=10, value=3)
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=4, value=1)

    if train_file and st.button("▶️ Arka Planda Eğitimi Başlat", type="primary"):
        suffix = ".csv" if train_file.name.endswith(".csv") else ".json"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(train_file.getvalue())
            tmp_path = tmp.name

        cmd = [
            sys.executable,
            "train.py",
            "--data",
            tmp_path,
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--output",
            "./trained_model",
        ]

        log_path = "train_log.txt"
        with open(log_path, "w", encoding="utf-8") as log:
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

        st.info(
            f"⏳ Eğitim arka planda başlatıldı.\n\n"
            f"• İlerlemeyi `{log_path}` dosyasından takip edebilirsiniz.\n"
            f"• Eğitim bitince sayfayı **yenileyin (F5)**.\n"
            f"• Eğitilmiş model otomatik olarak yüklenecektir."
        )