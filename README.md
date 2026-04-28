# 🎬 Project 2 — Movie Review Sentiment Analyzer

**Stack:** Python · NLTK · Scikit-learn · Streamlit  
**Algorithm:** Naive Bayes + TF-IDF Vectorizer  
**Dataset:** IMDB 50K Movie Reviews (Kaggle)  
**Deployment:** HuggingFace Spaces / Render

---

## 📁 Project Structure

```
project2-sentiment/
├── app.py              ← Streamlit UI (run this)
├── train_model.py      ← Train and save the model
├── requirements.txt    ← Dependencies
├── model.pkl           ← Saved after training
├── vectorizer.pkl      ← Saved after training
├── label_encoder.pkl   ← Saved after training
└── README.md
```

---

## ⚙️ Setup (Step by Step)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download dataset
1. Go to https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Download `IMDB Dataset.csv`
3. Place it in this folder (same folder as `train_model.py`)

### Step 3 — Train the model
```bash
python train_model.py
```
This creates `model.pkl`, `vectorizer.pkl`, `label_encoder.pkl`.  
Takes about 2–3 minutes. Do this only once.

### Step 4 — Run the app
```bash
streamlit run app.py
```

---

## 🚀 Deploy to HuggingFace Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Choose **Streamlit** as the SDK
3. Upload all files INCLUDING the `.pkl` model files
4. That's it — Spaces auto-installs from requirements.txt

> ⚠️ The `.pkl` files are large (~150MB). Use Git LFS on HuggingFace Spaces:
> ```bash
> git lfs install
> git lfs track "*.pkl"
> git add .gitattributes
> ```

---

## 🎯 Interview-Ready Explanations

### The Pipeline
```
Raw Review Text
      ↓
Preprocessing (HTML removal → lowercase → remove stopwords → stemming)
      ↓
TF-IDF Vectorizer (text → 50,000-dimension numeric vector)
      ↓
Multinomial Naive Bayes (predicts positive/negative + probability)
      ↓
Streamlit UI (shows result + confidence + word analysis)
```

### Key Concepts

| Concept | One-line explanation |
|---|---|
| **TF-IDF** | Gives high scores to words frequent in one doc but rare overall |
| **Naive Bayes** | Uses Bayes' theorem; assumes word independence given class |
| **Stemming** | Reduces words to root form (running → run) |
| **Stopwords** | Common words (the, is, at) that carry no sentiment signal |
| **Laplace smoothing** | Prevents zero probability for unseen words (alpha=0.1) |
| **Bigrams** | Two-word phrases ("not good") captured as features |
| **Accuracy** | ~86% on IMDB test set — typical for Naive Bayes + TF-IDF |

### Why Naive Bayes for NLP?
- Fast: trains in seconds on 40K reviews
- Handles sparse, high-dimensional data well (50K features)
- Works surprisingly well despite the "independence" assumption
- Industry baseline for text classification

### What is the "naive" assumption?
Naive Bayes assumes all words in a review are **independent of each other**
given the class label. In reality, "not good" is very different from
"not" and "good" separately — but the model treats them as independent.
Despite this simplification, it still achieves ~86% accuracy.

---

## 📊 Expected Results
- **Accuracy:** ~85–87%
- **F1-Score:** ~0.86
- Training time: ~2–3 minutes
- Inference time: <50ms per review

---

*Project 2 of 3 | TCS NQT Prime AI/ML Portfolio*
