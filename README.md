# 🚀 Semantic Recommendation System (Embeddings + ML Ranking)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-green.svg" />
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-orange.svg" />
  <img src="https://img.shields.io/badge/Embeddings-Sentence%20Transformers-purple.svg" />
  <img src="https://img.shields.io/badge/Status-Active-success.svg" />
</p>

---

## 📌 Overview

A **semantic recommendation system** that retrieves and ranks content using:

- 🧠 Sentence embeddings (semantic understanding)  
- ⚡ FAISS (fast vector search)  
- 📊 ML ranking (Logistic Regression over engineered features)

Unlike keyword-based systems, it surfaces relevant results even when there is **no direct word overlap**.

---

## ❓ Problem

Keyword-based systems fail to capture intent.

Example:

Query: "cooking show"

Desired results:

- MasterChef India  
- Street Food Stories  

❌ No keyword overlap → missed  
✅ Semantic retrieval solves this

---

## 💡 Solution

### Pipeline

Query → Embedding → FAISS → Candidate Retrieval → Feature Engineering → ML Ranking → Top-K Results

---

## 🏗️ Architecture

### 1️⃣ Embeddings
- Model: sentence-transformers/all-MiniLM-L6-v2  
- Converts text → dense vectors  
- Captures semantic meaning  

### 2️⃣ Retrieval (FAISS)
- Approximate nearest neighbor search  
- Retrieves top-N candidates  

### 3️⃣ Feature Engineering

Each candidate is scored using:

- Similarity score → semantic relevance  
- Category match → intent alignment  
- Lexical overlap → exact words (stopword filtered)  

### 4️⃣ ML Ranking

- Logistic Regression  
- Learns optimal feature weights  
- Outputs probability of relevance  

---

## 🔍 Example

Command:

python src/recommend.py "cooking show"

Output:

- Italian Cooking Masterclass  
- MasterChef India  
- Street Food Stories  
- Healthy Recipes  
- Vegan Cooking Guide  

---

## ⚙️ Key Improvements

- Fixed category leakage (irrelevant items appearing)  
- Removed lexical noise (e.g., "show", "series")  
- Added multi-intent query handling  
- Improved evaluation design  

---

## 📊 Evaluation

Metric: Precision@5

| Query | Precision@5 |
|------|------------|
| food travel show | 1.00 |
| kids learning videos | 1.00 |
| healthy lifestyle | 1.00 |
| family show | 0.00 |
| funny series | 0.00 |
| action adventure | 0.00 |

Average Precision@5 ≈ 0.50

---

## 📉 Observations

- Strong for structured queries  
- Weak for abstract or unseen queries  

---

## 🧠 Key Learnings

- Embeddings ≠ intent understanding  
- Feature engineering is critical  
- Labels define behavior  
- Evaluation can mislead  
- Candidate pool size matters  

---

## ⚠️ Limitations

- No personalization  
- Synthetic training data  
- Limited categories  
- No intent classification  
- No entity understanding  

---

## 🚀 Future Work

- Add personalization  
- Use learning-to-rank models  
- Expand dataset  
- Add intent detection  

---

## ▶️ How to Run

pip install -r requirements.txt  
python src/train_ranker.py  
python src/recommend.py "cooking show"  
python src/evaluate.py  

---

## 🛠️ Tech Stack

Python  
Sentence Transformers  
FAISS  
Scikit-learn  
NumPy  

---

## 🎯 Final Note

Built an embedding-based recommendation system using FAISS and ML ranking, improving Precision@5 through feature engineering and evaluation design.
Semantic similarity alone is not enough — combining multiple signals and proper evaluation is key.

---
