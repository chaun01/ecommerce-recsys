# FAISS Index for Fast Retrieval

This module provides FAISS (Facebook AI Similarity Search) integration for fast approximate nearest neighbor (ANN) search of items given user embeddings.

## Overview

FAISS enables efficient similarity search over large collections of embeddings. This is crucial for the retrieval stage where we need to quickly find candidate items from millions of possibilities.

## Supported Index Types

### 1. Flat (Exact Search)
- **Description**: Brute-force exact search
- **Pros**: Perfect accuracy
- **Cons**: Slow for large datasets (O(n))
- **Use case**: Small datasets (<100K items) or baseline

### 2. HNSW (Hierarchical Navigable Small World)
- **Description**: Graph-based approximate search
- **Pros**: Fast search, good accuracy, no training needed
- **Cons**: Higher memory usage
- **Use case**: Medium to large datasets (recommended for production)
- **Parameters**:
  - `m`: Number of connections (default: 32, higher = more accurate but slower)
  - `ef_construction`: Build quality (default: 40)
  - `ef_search`: Search quality (default: 16, higher = more accurate)

### 3. IVFFlat (Inverted File)
- **Description**: Clustering-based approximate search
- **Pros**: Lower memory than HNSW
- **Cons**: Requires training, slightly less accurate
- **Use case**: Large datasets where memory is limited
- **Parameters**:
  - `nlist`: Number of clusters (default: 100)
  - `nprobe`: Clusters to search (default: 10, higher = more accurate)

## Files

- `faiss_index.py`: FAISS index manager with build/search/save/load
- `build_index.py`: Script to build index from embeddings
- `test_faiss.py`: Test script to verify functionality

## Usage

### Building Index

```bash
# Build HNSW index (recommended)
python retrieval/build_index.py \
  --embeddings models/retrieval/checkpoints/item_embeddings.npy \
  --output retrieval/indices/item_index.faiss \
  --index_type HNSW \
  --m 32 \
  --ef_construction 40 \
  --ef_search 16 \
  --benchmark

# Build IVFFlat index
python retrieval/build_index.py \
  --embeddings models/retrieval/checkpoints/item_embeddings.npy \
  --output retrieval/indices/item_index_ivf.faiss \
  --index_type IVFFlat \
  --nlist 100 \
  --nprobe 10

# Build Flat index (exact search)
python retrieval/build_index.py \
  --embeddings models/retrieval/checkpoints/item_embeddings.npy \
  --output retrieval/indices/item_index_flat.faiss \
  --index_type Flat
```

### Benchmarking

```bash
# Benchmark with user embeddings
python retrieval/build_index.py \
  --embeddings models/retrieval/checkpoints/item_embeddings.npy \
  --output retrieval/indices/item_index.faiss \
  --index_type HNSW \
  --benchmark \
  --user_embeddings models/retrieval/checkpoints/user_embeddings.npy
```

### Using in Code

```python
from retrieval.faiss_index import FAISSIndex
import numpy as np

# Load index
index = FAISSIndex.load('retrieval/indices/item_index.faiss')

# Search for user
user_embedding = np.random.randn(1, 64).astype(np.float32)
top_items = index.search(user_embedding, k=100)

print(f"Top 100 candidate items: {top_items}")

# Search with distances
top_items, distances = index.search(user_embedding, k=100, return_distances=True)
print(f"Item IDs: {top_items}")
print(f"Similarity scores: {distances}")

# Get statistics
stats = index.get_stats()
print(stats)
```

### Batch Search

```python
# Search for multiple users at once
user_embeddings = np.random.randn(1000, 64).astype(np.float32)
top_items = index.search(user_embeddings, k=100)

print(f"Results shape: {top_items.shape}")  # (1000, 100)
```

## Performance Characteristics

### For 68K items dataset:

| Index Type | Build Time | Memory | Search (p95) | Recall@100 |
|------------|------------|--------|--------------|------------|
| Flat       | Instant    | ~17 MB | ~5 ms        | 100%       |
| HNSW       | ~30 sec    | ~25 MB | <1 ms        | 98-99%     |
| IVFFlat    | ~10 sec    | ~20 MB | ~2 ms        | 95-97%     |

*Note: Times are approximate and depend on hardware*

## Choosing Index Type

### Small dataset (<100K items)
- Use **Flat** for perfect accuracy
- Use **HNSW** if speed matters

### Medium dataset (100K - 1M items)
- Use **HNSW** (best balance)

### Large dataset (>1M items)
- Use **HNSW** if memory available
- Use **IVFFlat** if memory constrained

## Tuning Parameters

### HNSW
- **Accuracy ↑**: Increase `m` and `ef_search`
- **Speed ↑**: Decrease `ef_search`
- **Memory ↓**: Decrease `m`

### IVFFlat
- **Accuracy ↑**: Increase `nprobe`
- **Speed ↑**: Decrease `nprobe`
- **Memory ↓**: Decrease `nlist`

## Distance Metrics

### Inner Product (Recommended)
- For normalized embeddings (L2 norm = 1)
- Equivalent to cosine similarity
- Faster than L2 distance

### L2 Distance
- Euclidean distance
- Use if embeddings are not normalized

## Output Files

After building an index named `item_index.faiss`:
- `item_index.faiss`: FAISS index binary
- `item_index_item_ids.npy`: Item ID mapping
- `item_index_metadata.npy`: Index metadata
- `item_index_benchmark.yaml`: Benchmark results (if requested)

## Integration with Retrieval Model

The typical workflow is:

1. **Train Two-Tower model** → produces `item_embeddings.npy`
2. **Build FAISS index** → produces `item_index.faiss`
3. **Use in API**:
   ```python
   # Encode user
   user_emb = model.encode_user(user_history)

   # Retrieve candidates
   candidates = index.search(user_emb, k=100)

   # Rank candidates
   scores = ranking_model.score(user_emb, candidates)

   # Return top-K
   top_k = candidates[scores.argsort()[-10:]]
   ```

## Testing

```bash
# Run basic functionality test
python retrieval/test_faiss.py
```

## Troubleshooting

### "Index not trained" error
- IVFFlat requires training with data
- Ensure you have enough items (>= 30 * nlist)

### Slow search
- Increase `nprobe` (IVFFlat) or `ef_search` (HNSW)
- Use fewer items or smaller `k`

### Low recall
- Increase `nprobe` (IVFFlat) or `ef_search` (HNSW)
- Consider using Flat for exact search

### High memory usage
- Decrease `m` (HNSW) or `nlist` (IVFFlat)
- Use IVFFlat instead of HNSW
- Reduce embedding dimension

## References

- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
