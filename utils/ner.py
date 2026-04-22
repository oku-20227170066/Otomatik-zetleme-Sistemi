from transformers import pipeline

def load_ner():
    return pipeline(
        "ner",
        model="savasy/bert-base-turkish-ner-cased",
        aggregation_strategy="simple"
    )

def run_ner(text, ner_model):
    results = ner_model(text[:1500])

    entities = {"Kişi": set(), "Kurum": set(), "Yer": set()}
    mapping = {"PER": "Kişi", "ORG": "Kurum", "LOC": "Yer"}

    for r in results:
        label = mapping.get(r["entity_group"])
        if label:
            entities[label].add(r["word"])

    return {k: list(v) for k, v in entities.items()}