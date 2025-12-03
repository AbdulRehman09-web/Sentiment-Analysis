# 📊 **Sentiment Studio — Large CSV Friendly Sentiment Analysis App**

Sentiment Studio is a **Streamlit web application** built for **training and predicting sentiment** (Positive, Negative, Neutral) from large review datasets.
It supports **incremental training**, **stream processing**, and **model saving/loading**, making it ideal for datasets ranging from a few MB to over 1GB.

---

## 🚀 **Features**

### ✅ 1. **Built-in Dataset Support**

* The app loads `Reviews.csv` directly from the project folder (no upload required).
* Perfect for cloud deployment (Streamlit Cloud, HuggingFace Spaces, etc.).

### ✅ 2. **Handles Large CSV Files**

Supports:

* **Sample-based training** (fast, uses a subset)
* **Incremental “stream” training** (uses `SGDClassifier.partial_fit`)
* Works efficiently for **very large datasets** (>200MB).

### ✅ 3. **Real-Time Prediction**

* Clean and simple UI for predicting sentiment from a single input text.

### ✅ 4. **Model Saving / Loading**

* Save trained model & vectorizer as `.pkl`
* Upload `.pkl` models to reuse later

### ✅ 5. **Fully Automated Text Cleaning**

* URL removal
* HTML tag removal
* Punctuation removal
* Lowercasing
* Stopword-friendly cleaning

---

## 📁 **Project Structure**

```
├── streamlit_sentiment_large.py
├── Reviews.csv
├── large_sent_model.pkl         (generated after training)
├── large_sent_vectorizer.pkl    (generated after training)
├── README.md
```

---

## 🛠 **How It Works**

### **1️⃣ Load Data**

The app automatically loads:

```python
Reviews.csv
```

Make sure this file exists in the **same folder** as your Streamlit script.

You can override the path using:

```
Local CSV Path
```

---

### **2️⃣ Training Modes**

#### 🔹 *Sample Mode (Fast)*

* Loads a sample of rows
* Uses **TF-IDF** + **SGDClassifier**
* Good for quick training

#### 🔹 *Stream Mode (Memory Efficient)*

* Reads CSV in chunks
* Uses **HashingVectorizer**
* Incrementally trains with `partial_fit`
* Suitable for files >500MB+

---

## 📦 **Installation**

### Clone repo

```bash
git clone https://github.com/yourusername/sentiment-studio.git
cd sentiment-studio
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run streamlit_sentiment_large.py
```

---

## 🌐 **Deploy on Streamlit Cloud**

1. Upload your project to GitHub
2. Include `Reviews.csv` in the repo
3. Go to Streamlit Cloud → *Deploy App*
4. No local path required — the CSV loads automatically

---

## 📊 **Model Output Example**

After training, the model prints:

✔ Validation accuracy
✔ Classification report
✔ Saved model files
✔ Ready-to-use predictor

---

## 🎯 **Tech Stack**

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **HashingVectorizer / TfidfVectorizer**
* **SGDClassifier**

---

## 📝 **Future Enhancements**

* Add charts for sentiment distribution
* Add multi-language support
* Deploy pre-trained model version
* Add export predictions as CSV

---

## ❤️ **Author**

**Abdul Rehman**
AI Student | Data Scientist | ML Enthusiast