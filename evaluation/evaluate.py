"""
Evaluation script for recommendation system.

Evaluates the system on test data and generates metrics report.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Set
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import RecommendationMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_file: str) -> Dict[int, Set[int]]:
    """
    Load test data and create ground truth.

    Args:
        test_file: Path to test interactions file

    Returns:
        Dict mapping user_id to set of interacted items
    """
    logger.info(f"Loading test data from {test_file}")
    test_df = pd.read_parquet(test_file)

    ground_truth = {}
    for user_id, group in test_df.groupby('user_idx'):
        ground_truth[user_id] = set(group['item_idx'].values)

    logger.info(f"Loaded ground truth for {len(ground_truth):,} users")
    return ground_truth


def generate_mock_predictions(
    user_ids: list,
    num_items: int,
    k: int = 100
) -> Dict[int, list]:
    """
    Generate mock predictions for testing.

    Args:
        user_ids: List of user IDs
        num_items: Total number of items
        k: Number of predictions per user

    Returns:
        Dict mapping user_id to list of predicted items
    """
    logger.info("Generating mock predictions...")

    predictions = {}
    for user_id in tqdm(user_ids, desc="Generating predictions"):
        # Random predictions
        preds = np.random.choice(num_items, size=min(k, num_items), replace=False)
        predictions[user_id] = preds.tolist()

    return predictions


def evaluate_system(
    predictions: Dict[int, list],
    ground_truth: Dict[int, Set[int]],
    k_values: list = [5, 10, 20, 50, 100]
) -> Dict[str, float]:
    """
    Evaluate recommendation system.

    Args:
        predictions: Dict of user predictions
        ground_truth: Dict of ground truth items
        k_values: List of K values

    Returns:
        Dict of metrics
    """
    logger.info("Evaluating recommendations...")

    metrics = RecommendationMetrics(k_values=k_values)

    for user_id in tqdm(predictions.keys(), desc="Evaluating users"):
        if user_id in ground_truth:
            preds = predictions[user_id]
            gt = ground_truth[user_id]
            metrics.add_user(preds, gt)

    return metrics.get_metrics()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate recommendation system'
    )

    parser.add_argument(
        '--test_data',
        type=str,
        default='data/processed/test_interactions.parquet',
        help='Path to test interactions'
    )

    parser.add_argument(
        '--predictions',
        type=str,
        default=None,
        help='Path to predictions file (optional, will generate mock if not provided)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='evaluation/results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--k_values',
        type=int,
        nargs='+',
        default=[5, 10, 20, 50, 100],
        help='K values to evaluate'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Recommendation System Evaluation")
    print("="*80)

    # Load test data
    ground_truth = load_test_data(args.test_data)

    # Load or generate predictions
    if args.predictions and Path(args.predictions).exists():
        logger.info(f"Loading predictions from {args.predictions}")
        # TODO: Load actual predictions
        predictions = generate_mock_predictions(
            list(ground_truth.keys()),
            num_items=70000,
            k=100
        )
    else:
        logger.warning("No predictions file provided, generating mock predictions")
        predictions = generate_mock_predictions(
            list(ground_truth.keys()),
            num_items=70000,
            k=100
        )

    # Evaluate
    metrics = evaluate_system(predictions, ground_truth, args.k_values)

    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Number of users evaluated: {len(predictions):,}")
    print()

    for k in args.k_values:
        if f'recall@{k}' in metrics:
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

    # Save results
    results_file = output_dir / 'evaluation_results.csv'
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved results to {results_file}")

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
