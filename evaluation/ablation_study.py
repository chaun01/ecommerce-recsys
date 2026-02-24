"""
Ablation study comparing different pipeline configurations.

Compares:
1. Retrieval only (Two-Tower + FAISS)
2. Retrieval + Ranking (+ Wide & Deep)
3. Retrieval + Ranking + LLM Reranker (full pipeline)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import sys
import time

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


class AblationStudy:
    """Ablation study for recommendation pipeline."""

    def __init__(
        self,
        test_file: str,
        k_values: List[int] = [10, 20, 50]
    ):
        """
        Initialize ablation study.

        Args:
            test_file: Path to test data
            k_values: K values to evaluate
        """
        self.test_file = test_file
        self.k_values = k_values

        # Load test data
        logger.info(f"Loading test data from {test_file}")
        self.test_df = pd.read_parquet(test_file)

        # Build ground truth
        self.ground_truth = {}
        for user_id, group in self.test_df.groupby('user_idx'):
            self.ground_truth[user_id] = set(group['item_idx'].values)

        logger.info(f"Loaded {len(self.ground_truth):,} test users")

        # Get sample users for ablation (to save time)
        self.sample_users = list(self.ground_truth.keys())[:1000]
        logger.info(f"Using {len(self.sample_users):,} users for ablation study")

    def generate_mock_predictions(
        self,
        method: str,
        num_items: int = 70000
    ) -> Dict[int, List[int]]:
        """
        Generate mock predictions for different methods.

        In real implementation, this would use actual models.

        Args:
            method: Method name ('retrieval', 'ranking', 'llm')
            num_items: Total number of items

        Returns:
            Dict of predictions
        """
        predictions = {}

        # Simulate different quality levels
        if method == 'retrieval':
            # Lower quality - more random
            for user_id in self.sample_users:
                preds = np.random.choice(num_items, size=100, replace=False)
                predictions[user_id] = preds.tolist()

        elif method == 'ranking':
            # Medium quality - some overlap with ground truth
            for user_id in self.sample_users:
                gt = list(self.ground_truth.get(user_id, set()))
                # Mix 20% ground truth with 80% random
                if len(gt) > 0:
                    n_gt = min(20, len(gt))
                    gt_items = np.random.choice(gt, size=n_gt, replace=False)
                    random_items = np.random.choice(num_items, size=80, replace=False)
                    preds = np.concatenate([gt_items, random_items])
                    np.random.shuffle(preds)
                else:
                    preds = np.random.choice(num_items, size=100, replace=False)
                predictions[user_id] = preds[:100].tolist()

        elif method == 'llm':
            # Higher quality - more overlap
            for user_id in self.sample_users:
                gt = list(self.ground_truth.get(user_id, set()))
                # Mix 30% ground truth with 70% random
                if len(gt) > 0:
                    n_gt = min(30, len(gt))
                    gt_items = np.random.choice(gt, size=n_gt, replace=False)
                    random_items = np.random.choice(num_items, size=70, replace=False)
                    preds = np.concatenate([gt_items, random_items])
                    np.random.shuffle(preds)
                else:
                    preds = np.random.choice(num_items, size=100, replace=False)
                predictions[user_id] = preds[:100].tolist()

        return predictions

    def evaluate_configuration(
        self,
        config_name: str,
        predictions: Dict[int, List[int]]
    ) -> Dict[str, float]:
        """
        Evaluate one configuration.

        Args:
            config_name: Configuration name
            predictions: Predictions dict

        Returns:
            Metrics dict
        """
        logger.info(f"Evaluating {config_name}...")

        metrics = RecommendationMetrics(k_values=self.k_values)

        for user_id in self.sample_users:
            if user_id in predictions and user_id in self.ground_truth:
                preds = predictions[user_id]
                gt = self.ground_truth[user_id]
                metrics.add_user(preds, gt)

        return metrics.get_metrics()

    def run_ablation(self) -> pd.DataFrame:
        """
        Run ablation study.

        Returns:
            DataFrame with results
        """
        results = []

        # Configuration 1: Retrieval only
        logger.info("\n" + "="*80)
        logger.info("Configuration 1: Retrieval Only (Two-Tower + FAISS)")
        logger.info("="*80)

        start_time = time.time()
        retrieval_preds = self.generate_mock_predictions('retrieval')
        retrieval_time = (time.time() - start_time) / len(self.sample_users) * 1000

        retrieval_metrics = self.evaluate_configuration(
            'Retrieval Only',
            retrieval_preds
        )
        retrieval_metrics['avg_latency_ms'] = retrieval_time
        retrieval_metrics['configuration'] = 'Retrieval Only'
        results.append(retrieval_metrics)

        # Configuration 2: Retrieval + Ranking
        logger.info("\n" + "="*80)
        logger.info("Configuration 2: Retrieval + Ranking (+ Wide & Deep)")
        logger.info("="*80)

        start_time = time.time()
        ranking_preds = self.generate_mock_predictions('ranking')
        ranking_time = (time.time() - start_time) / len(self.sample_users) * 1000

        ranking_metrics = self.evaluate_configuration(
            'Retrieval + Ranking',
            ranking_preds
        )
        ranking_metrics['avg_latency_ms'] = ranking_time
        ranking_metrics['configuration'] = 'Retrieval + Ranking'
        results.append(ranking_metrics)

        # Configuration 3: Full Pipeline (+ LLM)
        logger.info("\n" + "="*80)
        logger.info("Configuration 3: Full Pipeline (+ LLM Reranker)")
        logger.info("="*80)

        start_time = time.time()
        llm_preds = self.generate_mock_predictions('llm')
        llm_time = (time.time() - start_time) / len(self.sample_users) * 1000

        llm_metrics = self.evaluate_configuration(
            'Full Pipeline',
            llm_preds
        )
        llm_metrics['avg_latency_ms'] = llm_time
        llm_metrics['configuration'] = 'Full Pipeline (+ LLM)'
        results.append(llm_metrics)

        # Create DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def print_comparison(self, results_df: pd.DataFrame):
        """Print comparison table."""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)
        print()

        # Select key metrics
        key_metrics = ['recall@10', 'ndcg@10', 'recall@20', 'ndcg@20', 'avg_latency_ms']

        # Create comparison table
        print(f"{'Configuration':<30}", end="")
        for metric in key_metrics:
            if metric == 'avg_latency_ms':
                print(f"{'Latency (ms)':>12}", end="")
            else:
                metric_name = metric.replace('_', ' ').replace('@', '@')
                print(f"{metric_name:>12}", end="")
        print()
        print("-" * 90)

        for _, row in results_df.iterrows():
            print(f"{row['configuration']:<30}", end="")
            for metric in key_metrics:
                if metric in row:
                    print(f"{row[metric]:>12.4f}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            print()

        print()

        # Calculate improvements
        if len(results_df) >= 2:
            print("Improvements over Retrieval Only:")
            print("-" * 80)

            baseline = results_df.iloc[0]

            for idx, row in results_df.iterrows():
                if idx == 0:
                    continue

                print(f"\n{row['configuration']}:")
                for metric in ['recall@10', 'ndcg@10']:
                    if metric in row and metric in baseline:
                        improvement = (row[metric] - baseline[metric]) / baseline[metric] * 100
                        print(f"  {metric}: {improvement:+.1f}%")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study'
    )

    parser.add_argument(
        '--test_data',
        type=str,
        default='data/processed/test_interactions.parquet',
        help='Path to test data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='evaluation/results/ablation_study.csv',
        help='Output file for results'
    )

    args = parser.parse_args()

    # Run ablation study
    study = AblationStudy(
        test_file=args.test_data,
        k_values=[10, 20, 50]
    )

    results_df = study.run_ablation()

    # Print results
    study.print_comparison(results_df)

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
