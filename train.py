import argparse
import os

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_data(path: str):
    """CSV veya JSON verisetini oku, HuggingFace Dataset'e çevir."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    else:
        raise ValueError("Sadece .csv veya .json dosyaları desteklenir.")

    if "text" not in df.columns or "summary" not in df.columns:
        raise ValueError("Verisetinde 'text' ve 'summary' sütunları zorunludur.")

    df = df[["text", "summary"]].dropna()
    return Dataset.from_pandas(df)


def preprocess_function(examples, tokenizer, max_input=512, max_target=128):
    """T5 formatında prefix + tokenization."""
    inputs = ["summarize: " + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Türkçe Özetleme Modeli Fine-tuning")
    parser.add_argument("--data", required=True, help="Eğitim veriseti (CSV/JSON)")
    parser.add_argument("--output", default="./trained_model", help="Model kayıt dizini")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (CPU için 1)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Öğrenme oranı")
    args = parser.parse_args()

    print(f"📂 Veriseti yükleniyor: {args.data}")
    dataset = load_data(args.data)
    print(f"✅ {len(dataset)} örnek yüklendi.")

    print("🤖 Tokenizer ve model yükleniyor (google/mt5-base)...")
    model_name = "google/mt5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("⚙️ Veriseti tokenize ediliyor...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=10,
        predict_with_generate=True,
        fp16=False,
        bf16=False,
        no_cuda=True,
        save_strategy="epoch",
        logging_dir=f"{args.output}/logs",
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Eğitim başlıyor... (CPU'da bu işlem uzun sürebilir)")
    trainer.train()

    print(f"💾 Model kaydediliyor: {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("🎉 Eğitim tamamlandı!")


if __name__ == "__main__":
    main()