"""
Script to build FAISS index from item embeddings.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.faiss_index import FAISSIndex, benchmark_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to build FAISS index."""
    parser = argparse.ArgumentParser(
        description='Build FAISS index from item embeddings'
    )

    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to item embeddings (.npy file)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save FAISS index'
    )

    parser.add_argument(
        '--index_type',
        type=str,
        default='HNSW',
        choices=['Flat', 'IVFFlat', 'HNSW'],
        help='Type of FAISS index (default: HNSW)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='inner_product',
        choices=['inner_product', 'l2'],
        help='Distance metric (default: inner_product)'
    )

    # IVFFlat parameters
    parser.add_argument(
        '--nlist',
        type=int,
        default=100,
        help='Number of clusters for IVFFlat (default: 100)'
    )

    parser.add_argument(
        '--nprobe',
        type=int,
        default=10,
        help='Number of clusters to search in IVFFlat (default: 10)'
    )

    # HNSW parameters
    parser.add_argument(
        '--m',
        type=int,
        default=32,
        help='Number of connections for HNSW (default: 32)'
    )

    parser.add_argument(
        '--ef_construction',
        type=int,
        default=40,
        help='Construction quality for HNSW (default: 40)'
    )

    parser.add_argument(
        '--ef_search',
        type=int,
        default=16,
        help='Search quality for HNSW (default: 16)'
    )

    parser.add_argument(
        '--item_mapping',
        type=str,
        default=None,
        help='Path to item mapping file (optional)'
    )

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark after building index'
    )

    parser.add_argument(
        '--user_embeddings',
        type=str,
        default=None,
        help='Path to user embeddings for benchmarking (optional)'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Building FAISS Index")
    logger.info("="*80)

    # Load item embeddings
    logger.info(f"Loading item embeddings from {args.embeddings}...")
    item_embeddings = np.load(args.embeddings)
    num_items, embedding_dim = item_embeddings.shape
    logger.info(f"Loaded {num_items:,} item embeddings with dimension {embedding_dim}")

    # Load item mapping if provided
    item_ids = None
    if args.item_mapping:
        logger.info(f"Loading item mapping from {args.item_mapping}...")
        item_mapping = pd.read_parquet(args.item_mapping)
        item_ids = item_mapping['item_idx'].values
        logger.info(f"Loaded {len(item_ids):,} item IDs")

    # Create FAISS index
    logger.info(f"Creating {args.index_type} index with {args.metric} metric...")
    index = FAISSIndex(
        embedding_dim=embedding_dim,
        index_type=args.index_type,
        metric=args.metric,
        nlist=args.nlist,
        nprobe=args.nprobe,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search
    )

    # Build index
    index.build_index(item_embeddings, item_ids)

    # Get statistics
    stats = index.get_stats()
    logger.info("Index statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Save index
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(output_path)

    # Benchmark if requested
    if args.benchmark:
        logger.info("="*80)
        logger.info("Running Benchmark")
        logger.info("="*80)

        if args.user_embeddings:
            logger.info(f"Loading user embeddings from {args.user_embeddings}...")
            user_embeddings = np.load(args.user_embeddings)
            logger.info(f"Loaded {len(user_embeddings):,} user embeddings")
        else:
            logger.info("No user embeddings provided, using random queries...")
            user_embeddings = np.random.randn(1000, embedding_dim).astype(np.float32)

        # Run benchmark
        results = benchmark_index(
            index=index,
            query_embeddings=user_embeddings,
            k_values=[10, 20, 50, 100],
            num_queries=min(1000, len(user_embeddings))
        )

        # Save benchmark results
        benchmark_path = output_path.parent / f"{output_path.stem}_benchmark.yaml"
        with open(benchmark_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        logger.info(f"Saved benchmark results to {benchmark_path}")

    logger.info("="*80)
    logger.info("FAISS Index Built Successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
