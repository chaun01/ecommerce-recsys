"""
Test script to verify FAISS index functionality.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from retrieval.faiss_index import FAISSIndex


def test_faiss_basic():
    """Test basic FAISS functionality."""
    print("Testing FAISS Index...")
    print("="*80)

    # Create synthetic data
    num_items = 1000
    num_queries = 10
    embedding_dim = 64

    print(f"Creating synthetic data:")
    print(f"  Items: {num_items}")
    print(f"  Queries: {num_queries}")
    print(f"  Embedding dim: {embedding_dim}")

    # Random embeddings
    np.random.seed(42)
    item_embeddings = np.random.randn(num_items, embedding_dim).astype(np.float32)
    query_embeddings = np.random.randn(num_queries, embedding_dim).astype(np.float32)

    # Test different index types
    for index_type in ["Flat", "HNSW", "IVFFlat"]:
        print(f"\n{'='*80}")
        print(f"Testing {index_type} index")
        print(f"{'='*80}")

        # Create index
        index = FAISSIndex(
            embedding_dim=embedding_dim,
            index_type=index_type,
            metric="inner_product"
        )

        # Build index
        print("Building index...")
        index.build_index(item_embeddings)

        # Get stats
        stats = index.get_stats()
        print(f"Index stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Search
        print(f"\nSearching for top-10 neighbors...")
        item_ids, distances = index.search(query_embeddings, k=10, return_distances=True)

        print(f"Results shape: {item_ids.shape}")
        print(f"Sample results (first query):")
        print(f"  Item IDs: {item_ids[0]}")
        print(f"  Distances: {distances[0]}")

        # Verify results
        assert item_ids.shape == (num_queries, 10), "Wrong result shape"
        print("[PASS] Test passed!")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_faiss_basic()
