import json
import joblib
import numpy as np
from embed import embed_texts
from index import VectorIndex

# Load data
with open("data/content.json") as f:
    items = json.load(f)

texts = [
    item["title"] + " " + item["description"]
    for item in items
]

# Step 1: Embed content
embeddings = embed_texts(texts)
lrmodel = joblib.load("model.pkl")
# Step 2: Build index
dim = embeddings.shape[1]
index = VectorIndex(dim)
index.add(embeddings)

STOPWORDS = {"show", "series", "video", "videos"}

def title_overlap(query, title):
    q_words = set(query.lower().split()) - STOPWORDS
    t_words = set(title.lower().split())

    overlap = q_words.intersection(t_words)

    return len(overlap) / max(len(q_words), 1)


def detect_categories(query):
    query = query.lower()

    categories = []

    if any(word in query for word in ["cook", "food", "recipe"]):
        categories.append("cooking")

    if any(word in query for word in ["gym", "fitness", "workout", "health"]):
        categories.append("fitness")

    if any(word in query for word in ["cartoon", "kids", "animation", "animated"]):
        categories.append("kids")

    if any(word in query for word in ["travel", "trip", "destination", "tour"]):
        categories.append("travel")

    return categories


def recommend(query, k=5):
    query_vec = embed_texts([query])
    D, I = index.search(query_vec, 20)

    query_categories = detect_categories(query)

    scored = []

    for rank, idx in enumerate(I[0]):
        item = items[idx]

        similarity = D[0][rank]

        # 🔥 category signal (only real signal you have)

        if not query_categories:
            category_match = 0.3
        else:
            category_match = 1.0 if item.get("category") in query_categories else 0.0

        overlap = title_overlap(query, item["title"])
        features = np.array([[similarity, category_match,overlap]])
        # 🔥 ML scoring
        final_score = lrmodel.predict_proba(features)[0][1]

        scored.append((final_score, item))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [item for _, item in scored[:k]]


if __name__ == "__main__":
    query = "Spartan: Ultimate Team Challenge"

    recs = recommend(query)

    print(f"\nQuery: {query}\n")
    print("Recommendations:\n")

    for r in recs:
        print(f"{r['title']} → {r['description']}")
