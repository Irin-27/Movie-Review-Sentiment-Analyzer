"""
app.py
======
Project 2 — Movie Review Sentiment Analyzer
Streamlit UI that loads the pre-trained model and predicts sentiment.

HOW TO RUN:
  streamlit run app.py
"""

import streamlit as st
import joblib
import re
import string
import numpy as np
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# NLTK Setup (runs once, silently)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

download_nltk()

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()

# ══════════════════════════════════════════════════════════════════════════════
# Load Models (cached — loaded only once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """Load pre-trained model, vectorizer, and label encoder."""
    required = ["model.pkl", "vectorizer.pkl", "label_encoder.pkl"]
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None
    model    = joblib.load("model.pkl")
    vec      = joblib.load("vectorizer.pkl")
    le       = joblib.load("label_encoder.pkl")
    return model, vec, le

model, vectorizer, label_encoder = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing (must match train_model.py exactly!)
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>",   " ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


def predict_sentiment(review: str):
    """Return (label_str, confidence, class_probabilities)."""
    cleaned  = clean_text(review)
    vec      = vectorizer.transform([cleaned])
    probs    = model.predict_proba(vec)[0]      # [P(neg), P(pos)]
    pred_idx = np.argmax(probs)
    label    = label_encoder.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx] * 100
    return label, confidence, probs

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown(
    "Enter any movie review below. "
    "The AI predicts whether it's **Positive** or **Negative** "
    "and shows a confidence score."
)
st.divider()

# ── Model not trained yet — show instructions ───────────────────────────────
if model is None:
    st.error("⚠️ Model files not found. Please run `train_model.py` first.")
    st.code("python train_model.py", language="bash")
    st.info(
        "1. Download IMDB Dataset from Kaggle  \n"
        "2. Place `IMDB Dataset.csv` in this folder  \n"
        "3. Run `python train_model.py`  \n"
        "4. Refresh this page"
    )
    st.stop()

# ── Review input ───────────────────────────────────────────────────────────────
review_input = st.text_area(
    label="✍️ Paste your movie review here:",
    placeholder=(
        "e.g. 'This movie was absolutely fantastic! "
        "The acting was superb and the story kept me hooked throughout.'"
    ),
    height=180,
    key="review_box",
)

col1, col2, col3 = st.columns([1, 1, 2])
analyze_btn  = col1.button("🔍 Analyze",   type="primary", use_container_width=True)
clear_btn    = col2.button("🗑️ Clear",    use_container_width=True)
example_btn  = col3.button("💡 Try an example", use_container_width=True)

# ── Example reviews ────────────────────────────────────────────────────────────
EXAMPLES = [
    "This film was a masterpiece! The direction, screenplay, and performances were all top-notch. I was completely mesmerized.",
    "Worst movie I've ever seen. Terrible acting, boring plot, and a complete waste of 2 hours. Don't bother.",
    "A decent film with some memorable scenes, though it drags in the second half. The lead performance saves it.",
]

if example_btn:
    import random
    # Store example in session so text_area shows it
    st.session_state["loaded_example"] = random.choice(EXAMPLES)
    st.rerun()

# Pre-fill text area with example if chosen
if "loaded_example" in st.session_state and not review_input:
    review_input = st.session_state.pop("loaded_example")
    # Rewrite the widget value by rerunning with prefilled key
    st.session_state["review_box"] = review_input
    st.rerun()

if clear_btn:
    st.session_state["review_box"] = ""
    st.rerun()

# ── Prediction output ──────────────────────────────────────────────────────────
if analyze_btn and review_input.strip():
    if len(review_input.strip()) < 10:
        st.warning("Please enter a longer review (at least 10 characters).")
    else:
        with st.spinner("Analyzing sentiment..."):
            label, confidence, probs = predict_sentiment(review_input)

        # Result card
        st.divider()
        is_positive = label == "positive"
        emoji  = "😊" if is_positive else "😞"
        color  = "🟢" if is_positive else "🔴"
        label_display = label.upper()

        st.markdown(f"### {emoji} Prediction: **{label_display}**")

        # Confidence bar
        st.markdown(f"**Confidence: {confidence:.1f}%**")
        st.progress(int(confidence))

        # Probability breakdown
        st.divider()
        st.markdown("#### 📊 Probability Breakdown")
        c1, c2 = st.columns(2)

        neg_pct = probs[0] * 100
        pos_pct = probs[1] * 100

        c1.metric(
            label="🔴 Negative",
            value=f"{neg_pct:.1f}%",
            delta=None,
        )
        c2.metric(
            label="🟢 Positive",
            value=f"{pos_pct:.1f}%",
            delta=None,
        )

        # Show what the model "sees" (top contributing words)
        st.divider()
        with st.expander("🔍 What words influenced the prediction?"):
            cleaned = clean_text(review_input)
            words   = cleaned.split()
            if words:
                # Get feature indices for words in the review
                vocab   = vectorizer.vocabulary_
                feature_log_prob = model.feature_log_prob_  # shape (2, n_features)
                # Difference in log-prob between positive and negative
                score_diff = feature_log_prob[1] - feature_log_prob[0]

                word_scores = []
                for w in set(words):
                    if w in vocab:
                        idx = vocab[w]
                        word_scores.append((w, score_diff[idx]))

                word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                top_words = word_scores[:10]

                if top_words:
                    for word, score in top_words:
                        direction = "Positive signal 🟢" if score > 0 else "Negative signal 🔴"
                        st.markdown(f"`{word}` → {direction} (score: {score:.3f})")
                else:
                    st.write("No recognizable words found in vocabulary.")
            else:
                st.write("Review was too short after preprocessing.")

elif analyze_btn and not review_input.strip():
    st.warning("Please enter a review before clicking Analyze.")

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — About & Interview Notes
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📚 About This Project")
    st.markdown("""
**Dataset:** IMDB 50K Movie Reviews  
**Algorithm:** Naive Bayes + TF-IDF  
**NLP Library:** NLTK  
**Framework:** Scikit-learn + Streamlit
""")
    st.divider()
    st.subheader("🎯 Interview Q&A")

    with st.expander("What is TF-IDF?"):
        st.write(
            "TF-IDF stands for Term Frequency × Inverse Document Frequency. "
            "TF measures how often a word appears in one document. "
            "IDF penalizes words that appear in almost every document (like 'the'). "
            "The product gives high scores to words that are common in one document "
            "but rare across the whole corpus — these are the most meaningful words."
        )

    with st.expander("Why Naive Bayes for text?"):
        st.write(
            "Naive Bayes works well for text because: (1) it's fast, "
            "(2) it handles high-dimensional sparse data (50k features) well, "
            "and (3) despite the 'naive' independence assumption, words in a "
            "review are roughly independent given the sentiment class, "
            "making the assumption not too far from reality."
        )

    with st.expander("What does 'naive' mean?"):
        st.write(
            "It assumes all features (words) are independent of each other "
            "given the class. In reality 'not good' is different from "
            "'not' + 'good' separately — that's the naivety. But even with "
            "this simplification, Naive Bayes achieves ~86% accuracy on IMDB."
        )

    with st.expander("What is Laplace smoothing?"):
        st.write(
            "If a word in the test set never appeared in training, "
            "P(word | class) = 0, making the whole probability 0 "
            "(zero-product problem). Laplace smoothing (alpha=0.1) adds a "
            "small constant to all word counts, so no probability is ever 0."
        )

    with st.expander("What is stemming?"):
        st.write(
            "Stemming reduces words to their root form. "
            "'running', 'runs', 'ran' all become 'run'. "
            "This reduces vocabulary size and lets the model generalize better. "
            "Porter Stemmer is the most common stemmer in NLP."
        )

    st.divider()
    st.caption("Project 2 of 3 | TCS NQT Prime AI/ML Portfolio")
