import streamlit as st
from transformers import pipeline


@st.cache_resource(show_spinner=False)
def load_ner_model():
    """Türkçe NER modelini cache'le (CPU)."""
    return pipeline(
        "ner",
        model="akdeniz27/bert-base-turkish-cased-ner",
        tokenizer="akdeniz27/bert-base-turkish-cased-ner",
        device=-1,
        aggregation_strategy="simple",
    )


def extract_entities(text: str):
    """
    Metinden PER (Kişi), ORG (Kurum), LOC (Yer) entity'lerini çıkar.
    Bitişik subword tokenlarını birleştirir, ## işaretlerini temizler.
    """
    if not text or not text.strip():
        return []

    model = load_ner_model()
    try:
        entities = model(text)
        allowed = {"PER", "ORG", "LOC"}
        
        # 1. Filtrele ve ## işaretlerini temizle
        cleaned = []
        for e in entities:
            ent_type = e.get("entity_group")
            if ent_type not in allowed:
                continue
            
            word = e["word"].replace("##", "").strip()
            if not word:
                continue
                
            cleaned.append({
                "word": word,
                "type": ent_type,
                "score": float(e["score"]),
                "start": e.get("start", 0),
                "end": e.get("end", 0),
            })
        
        # 2. Aynı tipte bitişik olanları birleştir
        # Örn: "DS" (end=102) + "Ö" (start=102) -> "DSÖ"
        merged = []
        for ent in cleaned:
            if merged and merged[-1]["type"] == ent["type"]:
                # Bitişik veya çok yakınsa (boşluk/## nedeniyle) birleştir
                if ent["start"] <= merged[-1]["end"] + 1:
                    merged[-1]["word"] += ent["word"]
                    merged[-1]["end"] = ent["end"]
                    merged[-1]["score"] = max(merged[-1]["score"], ent["score"])
                    continue
            merged.append(ent)
        
        # 3. Çok kısa (tek karakter) kalanları filtrele
        final = [m for m in merged if len(m["word"]) > 1]
        
        return [{"word": m["word"], "type": m["type"], "score": m["score"]} for m in final]
        
    except Exception as e:
        raise RuntimeError(f"NER işlemi sırasında hata: {str(e)}")
    #     streamlit run app.py --server.runOnSave true   