"""
train_model.py
==============
Project 2 — Movie Review Sentiment Analyzer
Run this ONCE to train and save the model.
After this, app.py loads the saved model (no re-training needed).

HOW TO RUN:
  1. Download the IMDB dataset from Kaggle:
     https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  2. Place 'IMDB Dataset.csv' in the same folder as this script.
  3. Run: python train_model.py
  4. This saves: model.pkl, vectorizer.pkl, label_encoder.pkl
"""

import pandas as pd
import numpy as np
import re
import string
import joblib
import os

# ── NLP libraries ──────────────────────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── ML libraries ───────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# ── Download NLTK data (one-time) ──────────────────────────────────────────────
print("📦 Downloading NLTK data...")
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Dataset
# ══════════════════════════════════════════════════════════════════════════════
print("\n📂 Loading dataset...")

CSV_PATH = "IMDB Dataset.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"'{CSV_PATH}' not found.\n"
        "Download it from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    )

df = pd.read_csv(CSV_PATH)
print(f"   Shape: {df.shape}")          # Should be (50000, 2)
print(f"   Columns: {list(df.columns)}")
print(f"   Sentiment counts:\n{df['sentiment'].value_counts()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Text Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
# INTERVIEW POINT:
#   Why preprocess?  Raw text is noisy. HTML tags, punctuation, and stopwords
#   add no sentiment signal but bloat the vocabulary.
#   Stemming reduces words to their root form (e.g. "loving" → "love"),
#   so "love" and "loved" map to the same feature.
# ──────────────────────────────────────────────────────────────────────────────

STOP_WORDS  = set(stopwords.words("english"))
stemmer     = PorterStemmer()

def clean_text(text: str) -> str:
    """Full preprocessing pipeline for a single review."""
    # 1. Remove HTML tags (IMDB reviews contain <br /> tags)
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 4. Remove punctuation and special characters — keep only letters
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Tokenize (split into words)
    tokens = text.split()

    # 6. Remove stopwords and very short words
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]

    # 7. Stemming — reduce to root form
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)


print("\n🧹 Cleaning reviews (this takes ~1–2 minutes)...")
df["clean_review"] = df["review"].apply(clean_text)
print("   ✅ Done!")

# Quick sanity check
print("\n   Sample before:", df["review"].iloc[0][:80])
print("   Sample after: ", df["clean_review"].iloc[0][:80])

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Encode Labels
# ══════════════════════════════════════════════════════════════════════════════
# positive → 1,  negative → 0
le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment"])   # positive=1, negative=0
print(f"\n🏷  Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train / Test Split
# ══════════════════════════════════════════════════════════════════════════════
X = df["clean_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 80% train, 20% test
    random_state=42,
    stratify=y            # Keep class balance in both splits
)
print(f"\n✂️  Train: {len(X_train)} | Test: {len(X_test)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TF-IDF Vectorization
# ══════════════════════════════════════════════════════════════════════════════
# INTERVIEW POINT:
#   TF-IDF = Term Frequency × Inverse Document Frequency
#   TF:  how often a word appears in THIS document
#   IDF: log(total_docs / docs_containing_word)  → penalizes common words
#   Result: rare-but-important words get higher scores
#
#   max_features=50000 means we keep only the top 50k words by TF-IDF score.
#   ngram_range=(1,2) means we consider both single words AND two-word phrases
#   (bigrams), e.g. "not good" is captured as a feature instead of just
#   "not" and "good" separately.
# ──────────────────────────────────────────────────────────────────────────────

print("\n📐 Building TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),      # unigrams + bigrams
    sublinear_tf=True,        # apply log(1+tf) — dampens effect of very frequent words
    min_df=5,                 # ignore words appearing in fewer than 5 docs
)

X_train_vec = vectorizer.fit_transform(X_train)   # fit + transform on train
X_test_vec  = vectorizer.transform(X_test)         # only transform on test (NO fit!)

print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")
print(f"   Matrix shape (train): {X_train_vec.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Train Naive Bayes
# ══════════════════════════════════════════════════════════════════════════════
# INTERVIEW POINT:
#   Multinomial Naive Bayes is ideal for TF-IDF features because:
#   (a) It works with non-negative counts/frequencies (TF-IDF values ≥ 0).
#   (b) It's extremely fast — O(n_features) per prediction.
#   (c) Despite its "naive" independence assumption it performs surprisingly
#       well on text (words are somewhat independent given the class).
#   alpha=0.1 is Laplace smoothing (prevents zero-probability for unseen words)
# ──────────────────────────────────────────────────────────────────────────────

print("\n🤖 Training Naive Bayes...")
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)
print("   ✅ Training complete!")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊 Evaluating on test set...")
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
print(f"\n   Accuracy: {acc * 100:.2f}%")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   {cm}")
print(f"   True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n💾 Saving model artifacts...")
joblib.dump(model,      "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le,         "label_encoder.pkl")
print("   Saved: model.pkl, vectorizer.pkl, label_encoder.pkl")
print("\n✅ Training complete! Run app.py to launch the UI.")
