# 🚀 Semantic Recommendation System (Embeddings + ML Ranking)

## 📌 Overview

This project implements a **semantic recommendation system** that retrieves and ranks content using:

- Sentence embeddings for semantic understanding  
- FAISS for fast vector search  
- Machine Learning (Logistic Regression) for ranking  

Unlike keyword-based systems, it can surface relevant results even when **no exact word match exists**.

---

## ❓ Problem

Traditional search systems rely on keyword matching and fail to capture intent.

Example:

Query: "cooking show"

Keyword systems struggle to match:

- MasterChef India  
- Street Food Stories  

because there is **no direct lexical overlap**.

---

## 💡 Solution

We use a **two-stage recommendation pipeline**:

Query → Embedding → FAISS → Candidate Retrieval → Feature Extraction → ML Ranking → Top-K Results

---

## 🏗️ Architecture

### 1️⃣ Embeddings
- Model: sentence-transformers/all-MiniLM-L6-v2  
- Converts text into dense vectors  
- Captures semantic meaning beyond keywords  

### 2️⃣ Candidate Retrieval (FAISS)
- Fast Approximate Nearest Neighbor search  
- Retrieves top-N candidates based on vector similarity  

### 3️⃣ Feature Engineering

Each candidate is scored using:

- Similarity Score → semantic relevance  
- Category Match → intent alignment  
- Lexical Overlap → exact word match (stopword filtered)  

### 4️⃣ ML Ranking

- Model: Logistic Regression  
- Learns optimal weighting of signals  
- Outputs probability of relevance  

---

## 🔍 Example

Query: "cooking show"

Recommendations:
- Italian Cooking Masterclass  
- MasterChef India  
- Street Food Stories  
- Healthy Recipes  
- Vegan Cooking Guide  

---

## ⚙️ Key Improvements & Iterations

- Fixed category leakage (cartoons appearing in cooking results)  
- Controlled lexical overlap using stopword filtering  
- Added multi-intent query handling  
- Improved evaluation to remove misleading perfect scores  

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

Average Precision@5: ~0.50

---

## 📉 Observations

- Performs well for structured and domain-aligned queries  
- Struggles with abstract or unseen queries  

---

## 🧠 Key Learnings

- Embeddings capture meaning, not intent  
- Feature engineering is critical  
- Labels define model behavior  
- Evaluation can be misleading  
- Candidate pool size impacts precision  

---

## ⚠️ Limitations

- No user personalization  
- Synthetic training data  
- Limited category coverage  
- No intent classification  
- No entity understanding  

---

## 🚀 Future Improvements

- Add personalization  
- Use learning-to-rank models  
- Expand dataset  
- Add intent classification  
- Use real user interaction data  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/train_ranker.py
python src/recommend.py "cooking show"
python src/evaluate.py
```

---

## 📁 Project Structure

semantic-recommender-system/
├── data/
│   └── content.json
├── src/
│   ├── embed.py
│   ├── index.py
│   ├── recommend.py
│   ├── train_ranker.py
│   ├── evaluate.py
├── model.pkl
├── requirements.txt
├── README.md

---

## 🛠️ Tech Stack

- Python  
- Sentence Transformers  
- FAISS  
- Scikit-learn  
- NumPy  

---

## 🎯 Final Note

Semantic similarity alone is not enough — combining signals and proper evaluation is key.

---
