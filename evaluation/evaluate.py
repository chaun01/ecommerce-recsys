"""
Offline evaluation and ablation study.
Compares: Retrieval only vs Retrieval + Ranking vs Retrieval + Ranking + LLM.
"""

import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import evaluate_recommendations
from retrieval.two_tower import TwoTowerModel
from retrieval.faiss_index import load_index, retrieve_top_k
from retrieval.train import compute_user_hist_emb
from ranking.wide_deep import WideAndDeep

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_VALUES = (5, 10, 20)
RETRIEVAL_N = 100


def load_pkl(name):
    with open(os.path.join(DATA_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def build_ground_truth(test_df):
    """Build ground truth: user -> set of interacted items in test."""
    gt = {}
    for uid, iid in zip(test_df["user_idx"], test_df["item_idx"]):
        gt.setdefault(uid, set()).add(iid)
    return gt


def evaluate_retrieval_only(user_embeddings, faiss_index, ground_truth, top_n=100):
    """Evaluate retrieval-only pipeline."""
    print("\n=== Retrieval Only ===")
    user_recs = {}

    users = list(ground_truth.keys())
    batch_size = 256
    for i in tqdm(range(0, len(users), batch_size), desc="Retrieval"):
        batch_users = users[i:i+batch_size]
        query_embs = user_embeddings[batch_users].astype(np.float32)
        _, indices = retrieve_top_k(faiss_index, query_embs, top_k=top_n)
        for j, uid in enumerate(batch_users):
            user_recs[uid] = indices[j].tolist()

    results = evaluate_recommendations(user_recs, ground_truth, K_VALUES)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    return results, user_recs


def evaluate_retrieval_ranking(user_embeddings, item_embeddings, faiss_index,
                                wide_deep, item_meta, user_histories, ground_truth,
                                top_n=100):
    """Evaluate retrieval + ranking pipeline."""
    print("\n=== Retrieval + Ranking ===")
    wide_deep.eval()
    cat_map = dict(zip(item_meta["item_idx"], item_meta["cat_idx"]))
    user_recs = {}

    users = list(ground_truth.keys())
    for uid in tqdm(users, desc="Retrieval+Ranking"):
        query_emb = user_embeddings[uid].reshape(1, -1).astype(np.float32)
        ret_scores, ret_indices = retrieve_top_k(faiss_index, query_emb, top_k=top_n)
        ret_scores = ret_scores[0]
        ret_indices = ret_indices[0]

        valid = ret_indices >= 0
        ret_scores = ret_scores[valid]
        ret_indices = ret_indices[valid]

        if len(ret_indices) == 0:
            user_recs[uid] = []
            continue

        with torch.no_grad():
            n = len(ret_indices)
            user_emb_t = torch.tensor(query_emb, dtype=torch.float32).expand(n, -1).to(DEVICE)
            item_ids_t = torch.tensor(ret_indices, dtype=torch.long).to(DEVICE)
            cat_ids_t = torch.tensor([cat_map.get(int(i), 0) for i in ret_indices],
                                      dtype=torch.long).to(DEVICE)
            ret_scores_t = torch.tensor(ret_scores, dtype=torch.float32).to(DEVICE)

            # Features
            user_hist = user_histories.get(uid, [])
            user_cats = [cat_map.get(h[0], 0) for h in user_hist]
            cat_counts = {}
            for c in user_cats:
                cat_counts[c] = cat_counts.get(c, 0) + 1

            evt_counts = torch.tensor(
                [min(cat_counts.get(cat_map.get(int(i), 0), 0), 50) / 50.0 for i in ret_indices],
                dtype=torch.float32
            ).to(DEVICE)
            recency = torch.ones(n, dtype=torch.float32).to(DEVICE)

            scores = wide_deep(user_emb_t, item_ids_t, cat_ids_t,
                               ret_scores_t, evt_counts, recency)
            scores = scores.cpu().numpy()

        sorted_idx = np.argsort(-scores)
        user_recs[uid] = ret_indices[sorted_idx].tolist()

    results = evaluate_recommendations(user_recs, ground_truth, K_VALUES)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    return results


def main():
    print(f"Device: {DEVICE}")

    # Load data
    test_df = load_pkl("test")
    item_meta = load_pkl("item_meta")
    user_histories = load_pkl("user_histories")
    n_items = load_pkl("n_items")
    n_cats = load_pkl("n_cats")

    ground_truth = build_ground_truth(test_df)
    print(f"Test users: {len(ground_truth)}")

    # Load models
    user_embeddings = np.load(os.path.join(MODEL_DIR, "user_embeddings.npy"))
    item_embeddings = np.load(os.path.join(MODEL_DIR, "item_embeddings.npy"))
    faiss_index = load_index()

    # Ablation 1: Retrieval only
    ret_results, _ = evaluate_retrieval_only(
        user_embeddings, faiss_index, ground_truth, RETRIEVAL_N
    )

    # Ablation 2: Retrieval + Ranking
    with open(os.path.join(MODEL_DIR, "wide_deep_config.pkl"), "rb") as f:
        wd_config = pickle.load(f)
    wide_deep = WideAndDeep(**wd_config).to(DEVICE)
    wide_deep.load_state_dict(torch.load(os.path.join(MODEL_DIR, "wide_deep.pt"),
                                          map_location=DEVICE, weights_only=True))

    rank_results = evaluate_retrieval_ranking(
        user_embeddings, item_embeddings, faiss_index,
        wide_deep, item_meta, user_histories, ground_truth, RETRIEVAL_N
    )

    # Summary
    lines = []
    lines.append("=" * 60)
    lines.append("ABLATION STUDY SUMMARY")
    lines.append("=" * 60)
    lines.append(f"{'Metric':<15} {'Retrieval':>12} {'Ret+Ranking':>12}")
    lines.append("-" * 39)
    for k in K_VALUES:
        r_key = f"Recall@{k}"
        n_key = f"NDCG@{k}"
        lines.append(f"{r_key:<15} {ret_results[r_key]:>12.4f} {rank_results[r_key]:>12.4f}")
        lines.append(f"{n_key:<15} {ret_results[n_key]:>12.4f} {rank_results[n_key]:>12.4f}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Note: LLM reranker evaluation requires OPENAI_API_KEY and is")
    lines.append("expensive to run on full test set. Use --llm flag for sampling.")

    summary = "\n".join(lines)
    print("\n" + summary)

    # Save to file
    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
