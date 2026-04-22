# 🚀 Türkçe NLP Analiz Sistemi

Modern doğal dil işleme (NLP) tekniklerini kullanarak Türkçe metinler üzerinde kapsamlı analiz yapmanı sağlayan bir web uygulamasıdır. Proje, **Streamlit arayüzü** ile kolay kullanım sunarken, arka planda güçlü Transformer modelleri çalıştırır.

---

## ✨ Özellikler

### 📝 Metin Özetleme

* **Abstractive (Yapay Zeka)**: Transformer tabanlı model ile yeni cümleler oluşturarak özet çıkarır
* **Extractive (Klasik)**: En önemli cümleleri seçerek özet oluşturur

### 🔍 Named Entity Recognition (NER)

Metin içerisinden otomatik olarak:

* 👤 Kişi isimleri
* 🏢 Kurum adları
* 📍 Yer bilgileri

tespit edilir.

### 📊 Görselleştirme

* Kelime frekansına dayalı **WordCloud** oluşturma

### 📏 Kalite Ölçümü

* ROUGE skorları ile özetlerin karşılaştırılması

  * ROUGE-1
  * ROUGE-L

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji    | Açıklama                |
| ------------ | ----------------------- |
| Streamlit    | Web arayüzü             |
| Transformers | NLP modelleri           |
| PyTorch      | Derin öğrenme altyapısı |
| NLTK         | Metin işleme            |
| WordCloud    | Görselleştirme          |
| Scikit-learn | TF-IDF işlemleri        |

---

## 📂 Proje Yapısı

```
turkce-nlp-pro/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── utils/
    ├── summarizer.py
    ├── ner.py
    └── visualization.py
```

---

## ⚙️ Kurulum

### 1. Depoyu klonla

```bash
git clone https://github.com/kullaniciadi/turkce-nlp-pro.git
cd turkce-nlp-pro
```

### 2. Sanal ortam oluştur (önerilir)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

---

## ▶️ Çalıştırma

```bash
streamlit run app.py
```

Tarayıcıda otomatik açılır:

```
http://localhost:8501
```

---

## 📥 Kullanım

1. Metni doğrudan gir veya PDF yükle
2. Sekmelerden istediğin analizi seç
3. Sonuçları anlık olarak görüntüle

---

## ⚡ Performans Notları

* İlk çalıştırmada modeller indirileceği için gecikme olabilir
* GPU varsa PyTorch otomatik hızlanır
* Uzun metinlerde işlem süresi artabilir

---

## 🚧 Geliştirme Fikirleri

* Daha büyük Türkçe LLM entegrasyonu
* Gerçek zamanlı NER görselleştirme
* API servisi (FastAPI) ekleme
* Çoklu dil desteği

---

## 🤝 Katkı

Katkıda bulunmak için:

1. Fork al
2. Yeni branch oluştur (`feature/...`)
3. Commit at
4. Pull Request gönder

---

## 📄 Lisans

MIT License

---

## 👨‍💻 Geliştirici

Bu proje, Türkçe NLP alanında pratik ve modern bir çözüm geliştirmek amacıyla hazırlanmıştır.