"""
FAISS index builder and retriever for approximate nearest neighbor search.
"""

import os
import numpy as np
import faiss
import pickle


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_faiss_index(item_embeddings, index_type="IVFFlat", nlist=100):
    """
    Build a FAISS index from item embeddings.

    Args:
        item_embeddings: (n_items, dim) numpy array, L2-normalized
        index_type: "Flat" for exact search, "IVFFlat" for approximate
        nlist: number of clusters for IVF index
    Returns:
        faiss.Index
    """
    dim = item_embeddings.shape[1]
    item_embeddings = item_embeddings.astype(np.float32)

    if index_type == "Flat":
        index = faiss.IndexFlatIP(dim)  # inner product (cosine since normalized)
        index.add(item_embeddings)
    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, len(item_embeddings)),
                                    faiss.METRIC_INNER_PRODUCT)
        index.train(item_embeddings)
        index.add(item_embeddings)
        index.nprobe = 10
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}, type={index_type}")
    return index


def save_index(index, path=None):
    if path is None:
        path = os.path.join(MODEL_DIR, "faiss_index.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    print(f"FAISS index saved to {path}")


def load_index(path=None):
    if path is None:
        path = os.path.join(MODEL_DIR, "faiss_index.bin")
    index = faiss.read_index(path)
    print(f"FAISS index loaded: {index.ntotal} vectors")
    return index


def retrieve_top_k(index, query_embeddings, top_k=100):
    """
    Retrieve top-K items for given query embeddings.

    Args:
        index: FAISS index
        query_embeddings: (n_queries, dim) numpy array, L2-normalized
        top_k: number of candidates to retrieve
    Returns:
        scores: (n_queries, top_k)
        indices: (n_queries, top_k)
    """
    query_embeddings = query_embeddings.astype(np.float32)
    scores, indices = index.search(query_embeddings, top_k)
    return scores, indices


def main():
    """Build and save FAISS index from exported item embeddings."""
    emb_path = os.path.join(MODEL_DIR, "item_embeddings.npy")
    item_embeddings = np.load(emb_path)
    print(f"Loaded item embeddings: {item_embeddings.shape}")

    # Use Flat for smaller datasets, IVFFlat for larger
    if item_embeddings.shape[0] < 50000:
        index = build_faiss_index(item_embeddings, index_type="Flat")
    else:
        index = build_faiss_index(item_embeddings, index_type="IVFFlat", nlist=256)

    save_index(index)
    print("FAISS index build complete!")


if __name__ == "__main__":
    main()
