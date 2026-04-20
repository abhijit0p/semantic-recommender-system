from recommend import recommend, detect_categories

STOPWORDS = {"show", "series", "video", "videos"}

def is_relevant(query, item, query_cats=None):
    # Avoid recomputing categories
    if query_cats is None:
        query_cats = detect_categories(query)

    title = item["title"].lower()

    # 🔹 1. Strong signal: category match
    if query_cats and item.get("category") in query_cats:
        return 1

    # 🔹 2. Fallback: lexical overlap (filtered)
    q_words = set(query.lower().split()) - STOPWORDS
    t_words = set(title.split())

    overlap = q_words.intersection(t_words)

    # Require at least 1 meaningful overlap word
    if overlap:
        return 1

    return 0


def precision_at_k(query, k=5):
    results = recommend(query, k)
    query_cats = detect_categories(query)

    relevant = 0
    for item in results:
        relevant += is_relevant(query, item, query_cats)

    return relevant / k


def run_evaluation():
    queries = [
    "family show",
    "funny series",
    "food travel show",
    "kids learning videos",
    "healthy lifestyle",
    "action adventure",
    ]

    k = 5

    print("\n--- Evaluation (Precision@5) ---\n")

    total = 0

    for q in queries:
        p = precision_at_k(q, k)
        total += p
        print(f"{q:20s} → {p:.2f}")

    avg = total / len(queries)

    print("\nAverage Precision@5:", round(avg, 2))


if __name__ == "__main__":
    run_evaluation()
