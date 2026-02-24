"""
FAISS Index Manager for fast approximate nearest neighbor search.

This module provides utilities to build and search FAISS indices for
retrieving candidate items given user embeddings.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS index manager for item retrieval."""

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "Flat",
        metric: str = "inner_product",
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 32,
        ef_construction: int = 40,
        ef_search: int = 16
    ):
        """
        Initialize FAISS index.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ('Flat', 'IVFFlat', 'HNSW')
            metric: Distance metric ('inner_product' or 'l2')
            nlist: Number of clusters for IVF (only for IVFFlat)
            nprobe: Number of clusters to search (only for IVFFlat)
            m: Number of connections for HNSW
            ef_construction: Construction time quality parameter (HNSW)
            ef_search: Search time quality parameter (HNSW)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index = None
        self.item_ids = None

    def build_index(self, embeddings: np.ndarray, item_ids: Optional[np.ndarray] = None):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Item embeddings of shape (num_items, embedding_dim)
            item_ids: Optional array of item IDs (if not provided, uses indices)
        """
        num_items, embedding_dim = embeddings.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embedding_dim}"
            )

        logger.info(f"Building {self.index_type} index for {num_items:,} items...")

        # Normalize embeddings for inner product search
        if self.metric == "inner_product":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Convert to float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)

        # Create index based on type
        if self.index_type == "Flat":
            # Exact search (brute force)
            if self.metric == "inner_product":
                self.index = faiss.IndexFlatIP(embedding_dim)
            else:
                self.index = faiss.IndexFlatL2(embedding_dim)

            self.index.add(embeddings)
            logger.info(f"Built Flat index with {self.index.ntotal:,} vectors")

        elif self.index_type == "IVFFlat":
            # Inverted file index for faster search
            if self.metric == "inner_product":
                quantizer = faiss.IndexFlatIP(embedding_dim)
                self.index = faiss.IndexIVFFlat(
                    quantizer, embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                quantizer = faiss.IndexFlatL2(embedding_dim)
                self.index = faiss.IndexIVFFlat(
                    quantizer, embedding_dim, self.nlist, faiss.METRIC_L2
                )

            # Train index
            logger.info(f"Training IVFFlat index with {self.nlist} clusters...")
            self.index.train(embeddings)

            # Add vectors
            self.index.add(embeddings)

            # Set number of clusters to search
            self.index.nprobe = self.nprobe

            logger.info(
                f"Built IVFFlat index with {self.index.ntotal:,} vectors, "
                f"nlist={self.nlist}, nprobe={self.nprobe}"
            )

        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            if self.metric == "inner_product":
                self.index = faiss.IndexHNSWFlat(embedding_dim, self.m, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexHNSWFlat(embedding_dim, self.m, faiss.METRIC_L2)

            # Set construction parameters
            self.index.hnsw.efConstruction = self.ef_construction

            # Add vectors
            self.index.add(embeddings)

            # Set search parameters
            self.index.hnsw.efSearch = self.ef_search

            logger.info(
                f"Built HNSW index with {self.index.ntotal:,} vectors, "
                f"M={self.m}, efConstruction={self.ef_construction}, efSearch={self.ef_search}"
            )

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Store item IDs
        if item_ids is not None:
            self.item_ids = item_ids
        else:
            self.item_ids = np.arange(num_items)

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 100,
        return_distances: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Search for top-k nearest neighbors.

        Args:
            query_embeddings: Query embeddings of shape (num_queries, embedding_dim)
            k: Number of neighbors to retrieve
            return_distances: Whether to return distances along with indices

        Returns:
            If return_distances=False:
                Item IDs of shape (num_queries, k)
            If return_distances=True:
                Tuple of (item_ids, distances), both of shape (num_queries, k)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Ensure query is 2D
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Normalize for inner product
        if self.metric == "inner_product":
            query_embeddings = query_embeddings / np.linalg.norm(
                query_embeddings, axis=1, keepdims=True
            )

        # Convert to float32
        query_embeddings = query_embeddings.astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embeddings, k)

        # Map indices to item IDs
        item_ids = self.item_ids[indices]

        if return_distances:
            return item_ids, distances
        else:
            return item_ids

    def save(self, path: Union[str, Path]):
        """
        Save index to disk.

        Args:
            path: Path to save index file
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path))

        # Save item IDs
        item_ids_path = path.parent / f"{path.stem}_item_ids.npy"
        np.save(item_ids_path, self.item_ids)

        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'm': self.m,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'num_items': len(self.item_ids)
        }

        metadata_path = path.parent / f"{path.stem}_metadata.npy"
        np.save(metadata_path, metadata)

        logger.info(f"Saved index to {path}")
        logger.info(f"Saved item IDs to {item_ids_path}")
        logger.info(f"Saved metadata to {metadata_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FAISSIndex':
        """
        Load index from disk.

        Args:
            path: Path to index file

        Returns:
            FAISSIndex instance
        """
        path = Path(path)

        # Load metadata
        metadata_path = path.parent / f"{path.stem}_metadata.npy"
        metadata = np.load(metadata_path, allow_pickle=True).item()

        # Create instance
        instance = cls(
            embedding_dim=metadata['embedding_dim'],
            index_type=metadata['index_type'],
            metric=metadata['metric'],
            nlist=metadata.get('nlist', 100),
            nprobe=metadata.get('nprobe', 10),
            m=metadata.get('m', 32),
            ef_construction=metadata.get('ef_construction', 40),
            ef_search=metadata.get('ef_search', 16)
        )

        # Load FAISS index
        instance.index = faiss.read_index(str(path))

        # Load item IDs
        item_ids_path = path.parent / f"{path.stem}_item_ids.npy"
        instance.item_ids = np.load(item_ids_path)

        logger.info(f"Loaded index from {path}")
        logger.info(f"Index contains {instance.index.ntotal:,} vectors")

        return instance

    def get_stats(self) -> dict:
        """Get index statistics."""
        if self.index is None:
            return {"status": "not_built"}

        stats = {
            "status": "built",
            "index_type": self.index_type,
            "metric": self.metric,
            "embedding_dim": self.embedding_dim,
            "num_items": self.index.ntotal,
            "memory_usage_mb": self._estimate_memory_usage()
        }

        if self.index_type == "IVFFlat":
            stats["nlist"] = self.nlist
            stats["nprobe"] = self.nprobe

        elif self.index_type == "HNSW":
            stats["m"] = self.m
            stats["ef_construction"] = self.ef_construction
            stats["ef_search"] = self.ef_search

        return stats

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        num_items = self.index.ntotal
        embedding_dim = self.embedding_dim

        # Base memory: embeddings
        base_memory = num_items * embedding_dim * 4 / (1024 * 1024)  # float32

        if self.index_type == "Flat":
            return base_memory

        elif self.index_type == "IVFFlat":
            # Additional memory for inverted lists
            return base_memory * 1.2

        elif self.index_type == "HNSW":
            # Additional memory for graph structure
            return base_memory * 1.5

        return base_memory


def create_index_from_config(config: dict) -> FAISSIndex:
    """
    Create FAISS index from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        FAISSIndex instance
    """
    index_config = config.get("index", {})

    return FAISSIndex(
        embedding_dim=config["model"]["embedding_dim"],
        index_type=index_config.get("type", "HNSW"),
        metric=index_config.get("metric", "inner_product"),
        nlist=index_config.get("nlist", 100),
        nprobe=index_config.get("nprobe", 10),
        m=index_config.get("m", 32),
        ef_construction=index_config.get("ef_construction", 40),
        ef_search=index_config.get("ef_search", 16)
    )


def benchmark_index(
    index: FAISSIndex,
    query_embeddings: np.ndarray,
    k_values: List[int] = [10, 20, 50, 100],
    num_queries: int = 1000
) -> dict:
    """
    Benchmark index search performance.

    Args:
        index: FAISS index
        query_embeddings: Query embeddings to test
        k_values: Different k values to test
        num_queries: Number of queries to run

    Returns:
        Dictionary with benchmark results
    """
    import time

    results = {}

    # Sample random queries
    num_available = min(num_queries, len(query_embeddings))
    query_indices = np.random.choice(len(query_embeddings), num_available, replace=False)
    queries = query_embeddings[query_indices]

    logger.info(f"Benchmarking index with {num_available} queries...")

    for k in k_values:
        # Warmup
        _ = index.search(queries[:10], k=k)

        # Benchmark
        start_time = time.time()
        _ = index.search(queries, k=k)
        elapsed = time.time() - start_time

        avg_latency = elapsed / num_available * 1000  # ms per query

        results[f"k={k}"] = {
            "total_time_s": elapsed,
            "avg_latency_ms": avg_latency,
            "qps": num_available / elapsed
        }

        logger.info(
            f"k={k}: {avg_latency:.2f}ms per query, {results[f'k={k}']['qps']:.2f} QPS"
        )

    return results
