🚀 Semantic Recommendation System using Embeddings + ML Ranking

📌 Overview

This project implements a semantic recommendation system that goes beyond keyword matching by leveraging:
Sentence embeddings for semantic understanding
FAISS for fast vector search
Machine learning for ranking
The system retrieves relevant content even when there is no direct keyword overlap.

❓ Problem

Traditional keyword-based systems fail to capture intent.

Example:
Query: "cooking show"

A keyword system struggles to match:
"MasterChef India"
"Street Food Stories" because there is no direct word overlap.

💡 Solution

This system uses a two-stage architecture:

Query → Embedding → FAISS → Candidate Retrieval → ML Ranking → Results

🏗️ Architecture
1. Embeddings (Semantic Understanding)
Model: sentence-transformers/all-MiniLM-L6-v2
Converts text into dense vectors

2. Retrieval (FAISS)
Efficient nearest-neighbor search
Retrieves top-N candidates based on vector similarity

3. Feature Engineering
Each candidate is scored using:
Similarity score (embedding distance)
Category match (intent alignment)
Lexical overlap (exact word match, stopword-filtered)

4. ML Ranking
Model: Logistic Regression
Learns optimal weighting of signals
Outputs probability of relevance

🔍 Example
Query: "cooking show"

Recommendations:
- Italian Cooking Masterclass
- MasterChef India
- Street Food Stories
- Healthy Recipes
- Vegan Cooking Guide

🧠 Key Improvements
✅ Multi-intent query handling
("food travel show" → cooking + travel)
✅ Stopword filtering
("show", "series" no longer distort overlap)
✅ Learned ranking instead of manual weights
✅ Larger candidate pool improves precision

📊 Evaluation
Metric: Precision@5

Query	Precision@5
food travel show	1.00
kids learning videos	1.00
healthy lifestyle	1.00
family show	0.40
funny series	0.00
action adventure	0.00
Average Precision@5: ~0.57

📉 Observations
Performs well for structured and domain-aligned queries
Struggles with:
abstract queries ("funny series")
unknown intents ("action adventure")

🧠 Key Learnings
Embeddings capture semantic similarity, not intent
Candidate pool size strongly impacts ranking quality
Labels define model behavior
Evaluation can be misleading if not designed carefully
Combining:
semantic signals
structured signals
lexical signals
→ gives best results

⚠️ Limitations
No user personalization
Synthetic training data (no real user clicks)
Limited category coverage
No intent classification layer
No entity understanding (e.g., "Spartan" event)

🚀 Future Improvements
Add user embeddings (personalization)
Replace logistic regression with learning-to-rank model
Add intent classification layer
Expand dataset and categories
Integrate real interaction data (CTR, clicks)

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train ranking model
python src/train_ranker.py
3. Run recommendations
python src/recommend.py
4. Run evaluation
python src/evaluate.py

📌 Tech Stack
Python
Sentence Transformers
FAISS
Scikit-learn
