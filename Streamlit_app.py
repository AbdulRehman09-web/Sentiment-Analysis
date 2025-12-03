# streamlit_sentiment_large_embedded.py
# Updated by ChatGPT: embeds a default CSV filename so you don't need to upload
# - Removed the CSV file_uploader option
# - Uses a default local filename (Reviews.csv) unless you change the path
# - Everything else kept functionally the same as your original script

import streamlit as st
import pandas as pd
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Sentiment Studio",
    page_icon="💬",
    layout="wide",
)

# -------------------------
# Helpers
# -------------------------

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|<.*?>|[^a-z\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def score_to_sentiment(x):
    try:
        v = float(x)
    except Exception:
        return None
    if v <= 2:
        return "negative"
    elif v == 3:
        return "neutral"
    else:
        return "positive"


def read_chunks(source, usecols, chunksize=200_000):
    # source: filepath or file-like object
    return pd.read_csv(source, usecols=usecols, chunksize=chunksize, iterator=True, dtype=str)


# -------------------------
# UI - Inputs
# -------------------------
st.title("Sentiment Studio — Large CSV Friendly 🚀")
st.write("This version expects a local CSV file. By default it looks for 'Reviews.csv' in the working directory.")

# Default embedded filename
DEFAULT_CSV = "Reviews.csv"

col1, col2 = st.columns([2, 1])
with col1:
    # Only local path input now — no upload widget
    local_path = st.text_input("Local CSV path (recommended for large files)", DEFAULT_CSV)
with col2:
    chunksize = st.number_input("Chunk size (rows)", value=200_000, min_value=1000, step=1000)
    mode = st.radio("Training mode", ("Sample (fast, needs sample in memory)", "Stream (incremental, memory-efficient)"))
    remove_neutral = st.checkbox("Remove neutral (score==3)", value=True)
    save_after = st.checkbox("Save trained model to disk", value=True)

st.markdown("---")
st.write("Specify column names (default: Text, Score).")
text_col = st.text_input("Text column name", value="Text")
score_col = st.text_input("Score column name", value="Score")
st.markdown("---")

# Determine source (now only local path)
source = local_path.strip() if local_path and local_path.strip() else None
if source is None:
    st.info("Provide a local path to your CSV to proceed.")
    st.stop()

# Preview small sample
st.info("Reading a small sample (first 1000 rows) for preview...")
try:
    sample_df = pd.read_csv(source, nrows=1000, usecols=[text_col, score_col], dtype=str)
except Exception as e:
    st.error(f"Failed to read sample: {e}")
    st.stop()

sample_df = sample_df.fillna("")
sample_df["Clean"] = sample_df[text_col].apply(clean_text)
sample_df["Sentiment"] = sample_df[score_col].apply(score_to_sentiment)
if remove_neutral:
    sample_df = sample_df[sample_df["Sentiment"] != "neutral"]
st.dataframe(sample_df.head(10))

# -------------------------
# Training
# -------------------------
if st.button("Start Training"):
    st.write("Training started...")

    # --- Sample mode ---
    if mode.startswith("Sample"):
        st.write("Sampling rows from file (keeps memory usage limited).")
        rows_to_collect = int(st.slider("Sample size (rows)", 5000, 200_000, 50_000, step=5000))
        texts, labels = [], []
        collected = 0
        try:
            for chunk in read_chunks(source, [text_col, score_col], chunksize=chunksize):
                chunk = chunk.fillna("")
                chunk["Sentiment"] = chunk[score_col].apply(score_to_sentiment)
                if remove_neutral:
                    chunk = chunk[chunk["Sentiment"] != "neutral"]
                if chunk.empty:
                    continue
                for t, s in zip(chunk[text_col].astype(str), chunk["Sentiment"]):
                    if s is None:
                        continue
                    texts.append(clean_text(t))
                    labels.append(s)
                    collected += 1
                    if collected >= rows_to_collect:
                        break
                if collected >= rows_to_collect:
                    break
        except Exception as e:
            st.error(f"Error while sampling: {e}")
            st.stop()

        if len(texts) < 100:
            st.error("Not enough samples collected to train. Try increasing sample size or check CSV.")
            st.stop()

        st.write(f"Collected {len(texts):,} samples. Vectorizing with TF-IDF and training SGDClassifier.")
        tf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X = tf.fit_transform(texts)
        clf = SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, tol=1e-3, random_state=42)

        Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=42)
        clf.fit(Xtr, ytr)
        preds = clf.predict(Xte)
        acc = accuracy_score(yte, preds)
        st.success(f"Sample-trained — validation accuracy: {acc:.4f}")
        st.text(classification_report(yte, preds))

        # store objects
        st.session_state["model"] = clf
        st.session_state["vectorizer"] = tf

    # --- Stream / Incremental mode ---
    else:
        st.write("Starting incremental training with HashingVectorizer + SGDClassifier.partial_fit")
        hv = HashingVectorizer(n_features=2 ** 18, alternate_sign=False, ngram_range=(1, 2))
        clf = SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1, tol=None, random_state=42)

        classes = np.array(["negative", "positive"])
        processed = 0
        val_texts, val_labels = [], []
        pbar = st.progress(0)
        max_val_samples = 20000

        try:
            reader = read_chunks(source, [text_col, score_col], chunksize=chunksize)
            for i, chunk in enumerate(reader):
                chunk = chunk.fillna("")
                chunk["Sentiment"] = chunk[score_col].apply(score_to_sentiment)
                if remove_neutral:
                    chunk = chunk[chunk["Sentiment"] != "neutral"]
                if chunk.empty:
                    continue
                texts_chunk = chunk[text_col].astype(str).apply(clean_text).tolist()
                labels_chunk = chunk["Sentiment"].tolist()
                X_chunk = hv.transform(texts_chunk)

                if processed == 0:
                    # first partial_fit: supply classes
                    clf.partial_fit(X_chunk, labels_chunk, classes=classes)
                else:
                    clf.partial_fit(X_chunk, labels_chunk)

                # collect validation examples (limited)
                if len(val_texts) < max_val_samples:
                    take = min(max_val_samples - len(val_texts), len(texts_chunk))
                    val_texts.extend(texts_chunk[:take])
                    val_labels.extend(labels_chunk[:take])

                processed += len(labels_chunk)
                # update progress (rough)
                # If you want a better progress estimate, adjust denominator
                pbar.progress(min(100, int(100 * processed / max(1, 1_000_000))))
        except StopIteration:
            pass
        except Exception as e:
            st.error(f"Streaming training failed: {e}")
            st.stop()

        st.success(f"Incremental training finished — processed approx {processed:,} rows.")
        st.session_state["model"] = clf
        st.session_state["vectorizer"] = hv

        if val_texts:
            Xv = hv.transform(val_texts)
            ypred = clf.predict(Xv)
            st.write("Incremental validation accuracy:", accuracy_score(val_labels, ypred))
            st.text(classification_report(val_labels, ypred))

    # Save model/vectorizer if desired
    if save_after and "model" in st.session_state and "vectorizer" in st.session_state:
        try:
            with open("large_sent_model.pkl", "wb") as mf:
                pickle.dump(st.session_state["model"], mf)
            with open("large_sent_vectorizer.pkl", "wb") as vf:
                pickle.dump(st.session_state["vectorizer"], vf)
            st.success("Saved large_sent_model.pkl and large_sent_vectorizer.pkl")
        except Exception as e:
            st.warning(f"Could not save model files: {e}")

# -------------------------
# Prediction block
# -------------------------
st.markdown("---")
st.header("Predict single review")
input_text = st.text_area("Enter review text to predict", height=150)
if st.button("Predict"):
    if not input_text.strip():
        st.warning("Enter text to predict.")
    elif "model" not in st.session_state or "vectorizer" not in st.session_state:
        st.error("No trained model in session. Train first or upload a model.")
    else:
        vec = st.session_state["vectorizer"].transform([clean_text(input_text)])
        pred = st.session_state["model"].predict(vec)[0]
        prob = None
        if hasattr(st.session_state["model"], "predict_proba"):
            prob = st.session_state["model"].predict_proba(vec).max()
        st.markdown(f"### Prediction: **{pred.upper()}**" + (f" — confidence: {prob:.2%}" if prob is not None else ""))

# -------------------------
# Upload pre-trained model
# -------------------------
# st.markdown("---")
# st.write("Or upload a pickled model (and vectorizer) to use for predictions.")
# model_upload = st.file_uploader("Upload model pickle (.pkl)", type=["pkl"], key="mup")
# vec_upload = st.file_uploader("Upload vectorizer pickle (.pkl)", type=["pkl"], key="vup")
# if model_upload is not None and vec_upload is not None:
#     try:
#         model_obj = pickle.load(model_upload)
#         vec_obj = pickle.load(vec_upload)
#         st.session_state["model"] = model_obj
#         st.session_state["vectorizer"] = vec_obj
#         st.success("Loaded model and vectorizer into session.")
#     except Exception as e:
#         st.error(f"Failed to load uploaded pickles: {e}")

# # Note for the user:
# # - Place your Reviews.csv in the same folder where you run `streamlit run streamlit_sentiment_large_embedded.py`
# # - Or change the Local CSV path field to point to the correct CSV file on disk
# # - This script no longer shows a CSV upload widget; it reads from the provided local path
