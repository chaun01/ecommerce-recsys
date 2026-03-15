"""
Dataset for ranking model training.
Generates (user, positive_item, negative_item) triplets with features.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset


class RankingDataset(Dataset):
    def __init__(self, interactions_df, user_histories, item_meta, user_embeddings,
                 item_embeddings, n_items, n_neg=4):
        """
        Args:
            interactions_df: DataFrame [user_idx, item_idx, weight, timestamp]
            user_histories: dict {user_idx: [(item_idx, weight, ts), ...]}
            item_meta: DataFrame [item_idx, cat_idx]
            user_embeddings: (n_users, dim) numpy array
            item_embeddings: (n_items, dim) numpy array
            n_items: total items
            n_neg: negatives per positive
        """
        self.n_items = n_items
        self.n_neg = n_neg
        self.user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32)
        self.item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)
        self.item_to_cat = dict(zip(item_meta["item_idx"], item_meta["cat_idx"]))

        # Build user -> interacted items set
        self.user_items = {}
        for uid, iid in zip(interactions_df["user_idx"], interactions_df["item_idx"]):
            self.user_items.setdefault(uid, set()).add(iid)

        # Build user -> category interaction counts
        self.user_cat_counts = {}
        for uid, iid in zip(interactions_df["user_idx"], interactions_df["item_idx"]):
            cat = self.item_to_cat.get(iid, 0)
            key = (uid, cat)
            self.user_cat_counts[key] = self.user_cat_counts.get(key, 0) + 1

        # Compute max timestamp per user for recency
        self.user_max_ts = interactions_df.groupby("user_idx")["timestamp"].max().to_dict()
        self.global_max_ts = interactions_df["timestamp"].max()

        # Build samples
        self.samples = list(
            zip(interactions_df["user_idx"].values,
                interactions_df["item_idx"].values,
                interactions_df["timestamp"].values)
        )

    def __len__(self):
        return len(self.samples)

    def _compute_features(self, user_idx, item_idx, ts):
        """Compute ranking features for a (user, item) pair."""
        user_emb = self.user_embeddings[user_idx]
        item_emb = self.item_embeddings[item_idx]

        # Retrieval score (cosine similarity)
        retrieval_score = torch.dot(user_emb, item_emb).item()

        # Category interaction count
        cat = self.item_to_cat.get(item_idx, 0)
        event_count = self.user_cat_counts.get((user_idx, cat), 0)
        event_count = min(event_count, 50) / 50.0  # normalize

        # Recency
        max_ts = self.user_max_ts.get(user_idx, self.global_max_ts)
        recency = 1.0 - (max_ts - ts) / (self.global_max_ts + 1)
        recency = max(0.0, min(1.0, recency))

        return cat, retrieval_score, event_count, recency

    def __getitem__(self, idx):
        user_idx, pos_item, ts = self.samples[idx]

        pos_cat, pos_ret_score, pos_evt_count, pos_recency = self._compute_features(
            user_idx, pos_item, ts
        )

        # Negative samples
        user_items = self.user_items.get(user_idx, set())
        neg_items = []
        while len(neg_items) < self.n_neg:
            neg = random.randint(0, self.n_items - 1)
            if neg not in user_items:
                neg_items.append(neg)

        neg_feats = [self._compute_features(user_idx, ni, ts) for ni in neg_items]

        # All candidates: [positive] + [negatives]
        all_items = [pos_item] + neg_items
        all_cats = [pos_cat] + [f[0] for f in neg_feats]
        all_ret_scores = [pos_ret_score] + [f[1] for f in neg_feats]
        all_evt_counts = [pos_evt_count] + [f[2] for f in neg_feats]
        all_recency = [pos_recency] + [f[3] for f in neg_feats]

        # Label: index 0 is positive
        labels = [1.0] + [0.0] * self.n_neg

        return {
            "user_idx": user_idx,
            "item_ids": torch.tensor(all_items, dtype=torch.long),
            "cat_ids": torch.tensor(all_cats, dtype=torch.long),
            "retrieval_scores": torch.tensor(all_ret_scores, dtype=torch.float32),
            "event_counts": torch.tensor(all_evt_counts, dtype=torch.float32),
            "recency": torch.tensor(all_recency, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }
