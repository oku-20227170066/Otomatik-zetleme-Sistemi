from models.extractive import extractive_summarize
from models.abstractive import abstractive_summarize
from models.ner import extract_entities
from models.visualization import generate_wordcloud
from rouge_score import rouge_scorer


def summarize_text(text: str, top_n: int = 3) -> dict:
    """
    Extractive + Abstractive özetleme, ROUGE, NER ve WordCloud.
    """
    if not text or not text.strip():
        raise ValueError("Giriş metni boş olamaz.")

    try:
        # 1. Extractive özet
        extractive = extractive_summarize(text, top_n=top_n)

        # 2. Abstractive özet
        abstractive = abstractive_summarize(text)

        # 3. ROUGE skorları
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(extractive, abstractive)
        rouge_results = {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

        # 4. NER
        entities = extract_entities(text)

        # 5. WordCloud
        wordcloud_fig = generate_wordcloud(text)

        return {
            "extractive": extractive,
            "abstractive": abstractive,
            "rouge": rouge_results,
            "entities": entities,
            "wordcloud": wordcloud_fig,
        }

    except Exception as e:
        raise RuntimeError(f"Özetleme sırasında hata oluştu: {str(e)}")