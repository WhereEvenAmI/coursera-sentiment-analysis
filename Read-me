# Coursera Reviews — Sentiment Analysis

Sentiment classification on 1.4M+ Coursera course reviews using two different NLP approaches: classical ML and deep learning. The goal was to compare how much improvement deep learning brings over traditional bag-of-words methods on the same dataset.

---

## Dataset

**Coursera Reviews Dataset** — available on Kaggle  
1,454,711 reviews across multiple courses, each with a star rating (1–5) converted to sentiment labels:
- `positive` — rating 4 or 5
- `neutral` — rating 3
- `negative` — rating 1 or 2

---

## Approaches

### 1. Classical ML (`bow_sentiment.ipynb`)

Text vectorization using **Bag of Words** and **TF-IDF**, classified with Logistic Regression.

| Method | Accuracy |
|---|---|
| Bag of Words + Logistic Regression | 96.3% |
| TF-IDF (unigram + bigram + trigram) | 95.6% |
| TF-IDF + Class Balancing | 88.6% |

Key finding: BoW outperformed TF-IDF on this dataset. Class balancing hurt overall accuracy but significantly improved recall on minority classes (negative, neutral).

---

### 2. Deep Learning (`coursera-project.ipynb`)

Sequence model using **GloVe (100d) embeddings** + **LSTM**, trained on Kaggle with GPU (Tesla T4).

| Method | Accuracy |
|---|---|
| GloVe + LSTM | 95.5% |

Architecture:
- Embedding layer (GloVe 100d, frozen weights, vocab size 20k)
- LSTM (128 units)
- Dropout (0.5)
- Dense (64, ReLU) → Dense (3, Softmax)

Trained for 5 epochs, batch size 128, ~77s per epoch on GPU.

---

## Results Summary

The classical BoW model (96.3%) marginally outperformed the LSTM (95.5%). This is partly explained by the heavily imbalanced dataset — 94% of reviews are positive — which makes accuracy a misleading metric. For the minority classes (negative, neutral), both models struggled, with LSTM showing slightly better recall.

---

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (CountVectorizer, TfidfVectorizer, LogisticRegression)
- TensorFlow / Keras (LSTM)
- GloVe pre-trained embeddings (Stanford NLP)
- Matplotlib, Seaborn
- Kaggle (GPU training)
