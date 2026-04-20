from recommend import recommend, detect_categories

def is_relevant(query, item):
    categories = detect_categories(query)

    # strong signal: category match
    if categories and item.get("category") in categories:
        return 1

    # fallback: lexical signal
    q_words = set(query.lower().split())
    t_words = set(item["title"].lower().split())

    overlap = q_words.intersection(t_words)

    return 1 if overlap else 0


def precision_at_k(query, k=5):
    results = recommend(query, k)
    categories  = detect_categories(query)

    relevant = 0

    for item in results:
        if is_relevant(query, item):
            relevant += 1

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