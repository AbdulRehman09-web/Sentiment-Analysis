# 📊 Sentiment-Analysis

A **Streamlit-based Sentiment Analysis App** designed specifically for **large CSV files (200MB+)**. This app supports **sample-based training**, **incremental streaming training**, and **real-time prediction** using scikit-learn models.

---

## 🚀 Features

### ✅ Large CSV Support (200MB+)

* Processes massive datasets using **chunking**.
* Two modes:

  * **Sample Mode** → Fast training using a fixed sample size.
  * **Stream Mode** → Memory-efficient incremental training.

### 🎯 Sentiment Classification

* Converts numerical ratings into:

  * **Positive**
  * **Negative**
  * **Neutral** (optional removal)

### 🧠 Machine Learning

* Uses **SGDClassifier** for scalable linear classification.
* Multiple vectorizers available:

  * **TF-IDF Vectorizer** (sample mode)
  * **Hashing Vectorizer** (stream mode)

### 📈 Evaluation

* Shows accuracy and classification report.
* Supports incremental validation.

### 🔍 Prediction

* Predict sentiment for a single text review.
* Option to load **pre-trained model + vectorizer**.

### 💾 Model Saving

* Automatically saves:

  * `large_sent_model.pkl`
  * `large_sent_vectorizer.pkl`

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2️⃣ Create virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

```bash
streamlit run streamlit_sentiment_large.py
```

---

## 🧩 Project Structure

```
📂 Sentiment-Analysis
│── streamlit_sentiment_large.py    # Main app
│── Reviews.csv                    # Your dataset
│── large_sent_model.pkl           # Saved model (optional)
│── large_sent_vectorizer.pkl      # Saved vectorizer (optional)
│── requirements.txt
└── README.md
```

---

## 📝 Usage Guide

### **Upload or specify CSV**

* Enter a **local file path** (recommended for 200MB+ files)
* Or upload CSV directly

### **Choose Training Mode**

* **Sample Mode** → Choose sample size (e.g., 50k rows)
* **Stream Mode** → Full dataset, incremental partial fitting

### **Train Model**

* Chunked reading
* Cleaning text
* Vectorization
* Model training
* Accuracy & report displayed

### **Predict Single Review**

Enter text → Get prediction + confidence score

---

## 📦 Requirements (from requirements.txt)

```
streamlit
pandas
numpy
scikit-learn
regex
pickleshare
```

---

## 📡 Model Files

After training, the following files are auto-created:

* `large_sent_model.pkl`
* `large_sent_vectorizer.pkl`

You can upload them back into the app anytime.

---

## 📘 Example Sentiments

### Positive

* "The product quality is amazing!"
* "I love this so much."
* "Highly recommended!"

### Negative

* "Terrible product, waste of money."
* "I’m disappointed with the quality."
* "Not worth buying at all."

---

## 🤝 Contributing

Feel free to fork this repository and submit pull requests.

---

## 🏷️ License

This project is open-source and available under the **MIT License**.

---

## ⭐ Support

If you like this project, consider giving it a **GitHub star** ⭐

---

### Developed by **Abdul Rehman**

