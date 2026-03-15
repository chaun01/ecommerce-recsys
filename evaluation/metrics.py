"""
Evaluation metrics for recommendation system.
Recall@K and NDCG@K.
"""

import numpy as np


def recall_at_k(recommended, relevant, k):
    """
    Recall@K: fraction of relevant items in top-K recommendations.

    Args:
        recommended: list of recommended item IDs (ordered)
        relevant: set of ground-truth relevant item IDs
        k: cutoff
    Returns:
        recall score
    """
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k & relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k):
    """
    Normalized Discounted Cumulative Gain @ K.

    Args:
        recommended: list of recommended item IDs (ordered)
        relevant: set of ground-truth relevant item IDs
        k: cutoff
    Returns:
        NDCG score
    """
    if not relevant:
        return 0.0

    # DCG
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because position starts at 1

    # Ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(user_recommendations, user_ground_truth, k_values=(5, 10, 20)):
    """
    Evaluate recommendation quality across all users.

    Args:
        user_recommendations: dict {user_idx: [item_idx, ...]} ordered recommendations
        user_ground_truth: dict {user_idx: set(item_idx, ...)} relevant items
        k_values: tuple of K values to evaluate
    Returns:
        dict of metrics
    """
    results = {}
    for k in k_values:
        recalls = []
        ndcgs = []
        for user_idx, recs in user_recommendations.items():
            relevant = user_ground_truth.get(user_idx, set())
            if not relevant:
                continue
            recalls.append(recall_at_k(recs, relevant, k))
            ndcgs.append(ndcg_at_k(recs, relevant, k))

        results[f"Recall@{k}"] = np.mean(recalls) if recalls else 0.0
        results[f"NDCG@{k}"] = np.mean(ndcgs) if ndcgs else 0.0

    return results
