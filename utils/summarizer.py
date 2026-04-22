import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def load_summarizer():
    model_name = "ozcangundes/mt5-small-turkish-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def run_abstractive(text, tokenizer, model):
    inputs = tokenizer("summarize: " + text,
                       return_tensors="pt",
                       truncation=True,
                       max_length=512)

    with torch.no_grad():
        output = model.generate(inputs["input_ids"],
                                max_length=150,
                                min_length=40)

    return tokenizer.decode(output[0], skip_special_tokens=True)

def run_extractive(text):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:5])