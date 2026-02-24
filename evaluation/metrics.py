"""
Evaluation metrics for recommendation systems.

Implements standard metrics:
- Recall@K
- Precision@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- Coverage
- Diversity
"""

import numpy as np
from typing import List, Set, Dict
from collections import defaultdict


def recall_at_k(predictions: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Calculate Recall@K.

    Recall@K = (# relevant items in top-K) / (# total relevant items)

    Args:
        predictions: List of predicted item IDs (ordered by relevance)
        ground_truth: Set of ground truth item IDs
        k: Cutoff position

    Returns:
        Recall@K score [0, 1]
    """
    if not ground_truth:
        return 0.0

    top_k = set(predictions[:k])
    hits = len(top_k & ground_truth)

    return hits / len(ground_truth)


def precision_at_k(predictions: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Calculate Precision@K.

    Precision@K = (# relevant items in top-K) / K

    Args:
        predictions: List of predicted item IDs
        ground_truth: Set of ground truth item IDs
        k: Cutoff position

    Returns:
        Precision@K score [0, 1]
    """
    if k == 0:
        return 0.0

    top_k = set(predictions[:k])
    hits = len(top_k & ground_truth)

    return hits / k


def dcg_at_k(predictions: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Calculate DCG@K (Discounted Cumulative Gain).

    DCG@K = sum(rel_i / log2(i+1)) for i in [1, K]

    Args:
        predictions: List of predicted item IDs
        ground_truth: Set of ground truth item IDs
        k: Cutoff position

    Returns:
        DCG@K score
    """
    dcg = 0.0

    for i, item_id in enumerate(predictions[:k], 1):
        if item_id in ground_truth:
            dcg += 1.0 / np.log2(i + 1)

    return dcg


def ndcg_at_k(predictions: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).

    NDCG@K = DCG@K / IDCG@K

    Args:
        predictions: List of predicted item IDs
        ground_truth: Set of ground truth item IDs
        k: Cutoff position

    Returns:
        NDCG@K score [0, 1]
    """
    if not ground_truth:
        return 0.0

    # Calculate DCG
    dcg = dcg_at_k(predictions, ground_truth, k)

    # Calculate IDCG (ideal DCG)
    ideal_predictions = list(ground_truth)[:k]
    idcg = dcg_at_k(ideal_predictions, ground_truth, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr(predictions: List[int], ground_truth: Set[int]) -> float:
    """
    Calculate MRR (Mean Reciprocal Rank).

    MRR = 1 / rank of first relevant item

    Args:
        predictions: List of predicted item IDs
        ground_truth: Set of ground truth item IDs

    Returns:
        MRR score [0, 1]
    """
    for i, item_id in enumerate(predictions, 1):
        if item_id in ground_truth:
            return 1.0 / i

    return 0.0


def average_precision(predictions: List[int], ground_truth: Set[int]) -> float:
    """
    Calculate Average Precision.

    AP = (sum of Precision@i for relevant items) / (# relevant items)

    Args:
        predictions: List of predicted item IDs
        ground_truth: Set of ground truth item IDs

    Returns:
        AP score [0, 1]
    """
    if not ground_truth:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, item_id in enumerate(predictions, 1):
        if item_id in ground_truth:
            hits += 1
            precision_sum += hits / i

    return precision_sum / len(ground_truth) if ground_truth else 0.0


def hit_rate_at_k(predictions: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Calculate Hit Rate@K.

    Hit Rate@K = 1 if any item in top-K is relevant, else 0

    Args:
        predictions: List of predicted item IDs
        ground_truth: Set of ground truth item IDs
        k: Cutoff position

    Returns:
        Hit rate [0, 1]
    """
    top_k = set(predictions[:k])
    return 1.0 if len(top_k & ground_truth) > 0 else 0.0


def coverage(all_predictions: List[List[int]], total_items: int) -> float:
    """
    Calculate catalog coverage.

    Coverage = (# unique items recommended) / (# total items)

    Args:
        all_predictions: List of prediction lists
        total_items: Total number of items in catalog

    Returns:
        Coverage [0, 1]
    """
    unique_items = set()
    for predictions in all_predictions:
        unique_items.update(predictions)

    return len(unique_items) / total_items if total_items > 0 else 0.0


def diversity_at_k(predictions: List[int], item_features: Dict[int, List], k: int) -> float:
    """
    Calculate diversity@K based on item features.

    Diversity = average pairwise distance between items in top-K

    Args:
        predictions: List of predicted item IDs
        item_features: Dictionary mapping item_id to feature vector
        k: Cutoff position

    Returns:
        Diversity score [0, 1]
    """
    top_k = predictions[:k]

    if len(top_k) < 2:
        return 0.0

    # Calculate pairwise distances
    distances = []
    for i in range(len(top_k)):
        for j in range(i + 1, len(top_k)):
            item_i = top_k[i]
            item_j = top_k[j]

            if item_i in item_features and item_j in item_features:
                feat_i = np.array(item_features[item_i])
                feat_j = np.array(item_features[item_j])

                # Cosine distance
                distance = 1 - np.dot(feat_i, feat_j) / (
                    np.linalg.norm(feat_i) * np.linalg.norm(feat_j) + 1e-9
                )
                distances.append(distance)

    return np.mean(distances) if distances else 0.0


class RecommendationMetrics:
    """Aggregated metrics for recommendation evaluation."""

    def __init__(self, k_values: List[int] = [5, 10, 20, 50]):
        """
        Initialize metrics calculator.

        Args:
            k_values: List of K values to evaluate
        """
        self.k_values = k_values
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
        self.num_users = 0

    def add_user(
        self,
        predictions: List[int],
        ground_truth: Set[int]
    ):
        """
        Add metrics for one user.

        Args:
            predictions: Predicted item IDs
            ground_truth: Ground truth item IDs
        """
        self.num_users += 1

        # Calculate metrics for each K
        for k in self.k_values:
            self.metrics[f'recall@{k}'].append(recall_at_k(predictions, ground_truth, k))
            self.metrics[f'precision@{k}'].append(precision_at_k(predictions, ground_truth, k))
            self.metrics[f'ndcg@{k}'].append(ndcg_at_k(predictions, ground_truth, k))
            self.metrics[f'hit_rate@{k}'].append(hit_rate_at_k(predictions, ground_truth, k))

        # Metrics without K
        self.metrics['mrr'].append(mrr(predictions, ground_truth))
        self.metrics['map'].append(average_precision(predictions, ground_truth))

    def get_metrics(self) -> Dict[str, float]:
        """
        Get average metrics across all users.

        Returns:
            Dictionary of metric names to average values
        """
        if self.num_users == 0:
            return {}

        avg_metrics = {}
        for metric_name, values in self.metrics.items():
            avg_metrics[metric_name] = np.mean(values)

        return avg_metrics

    def print_metrics(self):
        """Print metrics in a readable format."""
        metrics = self.get_metrics()

        print("\n" + "="*80)
        print("Recommendation Metrics")
        print("="*80)
        print(f"Number of users: {self.num_users:,}")
        print()

        # Group by K
        for k in self.k_values:
            print(f"K = {k}:")
            print(f"  Recall@{k}:    {metrics[f'recall@{k}']:.4f}")
            print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            print(f"  NDCG@{k}:      {metrics[f'ndcg@{k}']:.4f}")
            print(f"  Hit Rate@{k}:  {metrics[f'hit_rate@{k}']:.4f}")
            print()

        print("Overall:")
        print(f"  MRR: {metrics['mrr']:.4f}")
        print(f"  MAP: {metrics['map']:.4f}")
        print("="*80)


def evaluate_recommendations(
    predictions_dict: Dict[int, List[int]],
    ground_truth_dict: Dict[int, Set[int]],
    k_values: List[int] = [5, 10, 20, 50]
) -> Dict[str, float]:
    """
    Evaluate recommendations for multiple users.

    Args:
        predictions_dict: Dict mapping user_id to list of predicted items
        ground_truth_dict: Dict mapping user_id to set of ground truth items
        k_values: List of K values to evaluate

    Returns:
        Dictionary of average metrics
    """
    metrics = RecommendationMetrics(k_values=k_values)

    for user_id, predictions in predictions_dict.items():
        if user_id in ground_truth_dict:
            ground_truth = ground_truth_dict[user_id]
            metrics.add_user(predictions, ground_truth)

    return metrics.get_metrics()
