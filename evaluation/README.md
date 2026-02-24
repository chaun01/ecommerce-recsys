# Evaluation Module

Comprehensive evaluation framework for recommendation system with standard metrics and ablation studies.

## Metrics Implemented

### Recall@K
Measures coverage of relevant items in top-K recommendations.
```
Recall@K = (# relevant items in top-K) / (# total relevant items)
```

### Precision@K
Measures accuracy of top-K recommendations.
```
Precision@K = (# relevant items in top-K) / K
```

### NDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality with position discount.
```
NDCG@K = DCG@K / IDCG@K
DCG@K = sum(rel_i / log2(i+1))
```

### MRR (Mean Reciprocal Rank)
Measures rank of first relevant item.
```
MRR = 1 / (rank of first relevant item)
```

### MAP (Mean Average Precision)
Average precision across all relevant items.

### Hit Rate@K
Binary metric: 1 if any relevant item in top-K, else 0.

## Quick Start

### Run Evaluation

```bash
# Evaluate on test data
python evaluation/evaluate.py \
  --test_data data/processed/test_interactions.parquet \
  --k_values 5 10 20 50

# With predictions file
python evaluation/evaluate.py \
  --test_data data/processed/test_interactions.parquet \
  --predictions evaluation/predictions.npy \
  --output evaluation/results
```

### Run Ablation Study

```bash
python evaluation/ablation_study.py \
  --test_data data/processed/test_interactions.parquet \
  --output evaluation/results/ablation_study.csv
```

## Usage in Code

### Evaluate Single User

```python
from evaluation.metrics import recall_at_k, ndcg_at_k

predictions = [101, 203, 405, 506, 708]  # Predicted items
ground_truth = {203, 405, 999}           # Actual items

recall = recall_at_k(predictions, ground_truth, k=10)
ndcg = ndcg_at_k(predictions, ground_truth, k=10)

print(f"Recall@10: {recall:.4f}")
print(f"NDCG@10: {ndcg:.4f}")
```

### Evaluate Multiple Users

```python
from evaluation.metrics import RecommendationMetrics

metrics = RecommendationMetrics(k_values=[5, 10, 20])

for user_id, predictions in all_predictions.items():
    ground_truth = all_ground_truth[user_id]
    metrics.add_user(predictions, ground_truth)

results = metrics.get_metrics()
metrics.print_metrics()
```

### Full Evaluation

```python
from evaluation.metrics import evaluate_recommendations

predictions_dict = {
    user1: [item1, item2, ...],
    user2: [item3, item4, ...],
}

ground_truth_dict = {
    user1: {item1, item5, ...},
    user2: {item3, item7, ...},
}

metrics = evaluate_recommendations(
    predictions_dict,
    ground_truth_dict,
    k_values=[5, 10, 20]
)

print(f"Recall@10: {metrics['recall@10']:.4f}")
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
```

## Ablation Study

Compares 3 configurations:

1. **Retrieval Only** (Two-Tower + FAISS)
   - Fast (~1ms)
   - Moderate accuracy

2. **Retrieval + Ranking** (+ Wide & Deep)
   - Medium speed (~15ms)
   - Better accuracy

3. **Full Pipeline** (+ LLM Reranker)
   - Slower (~60ms)
   - Best accuracy

### Example Output

```
ABLATION STUDY RESULTS
================================================================================

Configuration                  Recall@10    NDCG@10    Recall@20    NDCG@20    Latency (ms)
----------------------------------------------------------------------------------------------
Retrieval Only                     0.1234      0.0856      0.2156      0.1234         1.2
Retrieval + Ranking                0.1678      0.1145      0.2845      0.1678        15.4
Full Pipeline (+ LLM)              0.1845      0.1289      0.3123      0.1845        58.7

Improvements over Retrieval Only:
--------------------------------------------------------------------------------

Retrieval + Ranking:
  recall@10: +36.0%
  ndcg@10: +33.8%

Full Pipeline (+ LLM):
  recall@10: +49.5%
  ndcg@10: +50.6%
```

## Evaluation Workflow

```bash
# 1. Train models
python models/retrieval/train_retrieval.py
python retrieval/build_index.py --embeddings ...

# 2. Generate predictions on test set
python scripts/generate_predictions.py \
  --test_data data/processed/test_interactions.parquet \
  --output evaluation/predictions.npy

# 3. Evaluate
python evaluation/evaluate.py \
  --test_data data/processed/test_interactions.parquet \
  --predictions evaluation/predictions.npy

# 4. Run ablation study
python evaluation/ablation_study.py
```

## Expected Performance

### Baseline (Random)
- Recall@10: ~0.001
- NDCG@10: ~0.0005

### Retrieval Only
- Recall@10: ~0.10-0.15
- NDCG@10: ~0.08-0.12

### Retrieval + Ranking
- Recall@10: ~0.15-0.20
- NDCG@10: ~0.12-0.18

### Full Pipeline
- Recall@10: ~0.18-0.25
- NDCG@10: ~0.15-0.22

*Note: Actual numbers depend on data quality and model training*

## Metrics Interpretation

### Recall@K
- **Good**: > 0.15
- **Fair**: 0.10 - 0.15
- **Poor**: < 0.10

Higher = more relevant items found

### NDCG@K
- **Good**: > 0.20
- **Fair**: 0.10 - 0.20
- **Poor**: < 0.10

Higher = better ranking quality

### MRR
- **Good**: > 0.30
- **Fair**: 0.15 - 0.30
- **Poor**: < 0.15

Higher = relevant items ranked higher

## Output Files

After running evaluation:

```
evaluation/
└── results/
    ├── evaluation_results.csv       # Detailed metrics
    ├── ablation_study.csv           # Ablation comparison
    └── metrics_summary.txt          # Human-readable summary
```

## Customization

### Add Custom Metric

```python
# In metrics.py

def custom_metric(predictions, ground_truth, k):
    # Your metric implementation
    return score

# In RecommendationMetrics.add_user()
self.metrics['custom'].append(custom_metric(predictions, ground_truth, k))
```

### Change K Values

```python
metrics = RecommendationMetrics(k_values=[3, 5, 10, 15, 20, 50, 100])
```

### Filter Users

```python
# Only evaluate users with >= N interactions
min_interactions = 5
filtered_users = [
    user for user, gt in ground_truth.items()
    if len(gt) >= min_interactions
]
```

## Comparison with Baselines

```python
# Random baseline
random_metrics = evaluate_random_baseline(test_data, k=10)

# Popularity baseline
popularity_metrics = evaluate_popularity_baseline(test_data, k=10)

# Your model
model_metrics = evaluate_model(model, test_data, k=10)

# Compare
print("Improvement over random:")
print(f"  NDCG@10: {(model_metrics['ndcg@10'] / random_metrics['ndcg@10'] - 1) * 100:.1f}%")
```

## Statistical Significance

```python
from scipy import stats

# Bootstrap confidence intervals
def bootstrap_metric(predictions, ground_truth, metric_fn, n_bootstrap=1000):
    scores = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(predictions), len(predictions))
        sample_scores = [metric_fn(predictions[i], ground_truth[i]) for i in indices]
        scores.append(np.mean(sample_scores))

    return np.percentile(scores, [2.5, 97.5])

ci = bootstrap_metric(predictions, ground_truth, lambda p, g: ndcg_at_k(p, g, 10))
print(f"95% CI for NDCG@10: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

## Best Practices

1. **Always use time-based split** for test data
2. **Report multiple metrics** (Recall, NDCG, MRR)
3. **Test on multiple K values** (5, 10, 20, 50)
4. **Run ablation studies** to understand contributions
5. **Compare with baselines** (random, popularity)
6. **Check statistical significance**
7. **Monitor latency** alongside accuracy

## Troubleshooting

### Metrics are 0 or very low
- Check ground truth is not empty
- Verify predictions are in correct format
- Ensure item IDs match between predictions and ground truth

### Slow evaluation
- Sample subset of users
- Parallelize with multiprocessing
- Reduce number of K values

### Memory issues
- Process users in batches
- Don't store all predictions in memory
- Use generators

## References

- Recall/Precision: Standard IR metrics
- NDCG: Järvelin & Kekäläinen (2002)
- MRR: Voorhees (1999)
- MAP: Baeza-Yates & Ribeiro-Neto (1999)
