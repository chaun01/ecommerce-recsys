"""
Dataset for Two-Tower model training.
Each sample: (user_history_embedding, positive_item, negative_items).
"""

import random
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    def __init__(self, interactions_df, user_histories, item_meta, n_items, max_hist_len=50, n_neg=4):
        """
        Args:
            interactions_df: DataFrame with columns [user_idx, item_idx, weight, timestamp]
            user_histories: dict {user_idx: [(item_idx, weight, ts), ...]}
            item_meta: DataFrame with [item_idx, cat_idx]
            n_items: total number of items
            max_hist_len: max history items to consider
            n_neg: number of negative samples per positive
        """
        self.n_items = n_items
        self.n_neg = n_neg
        self.max_hist_len = max_hist_len

        # Build item -> cat mapping
        self.item_to_cat = dict(zip(item_meta["item_idx"], item_meta["cat_idx"]))

        # Build set of all items for negative sampling
        self.all_items = set(range(n_items))

        # Build samples: for each interaction, store (user_idx, pos_item, timestamp)
        self.samples = list(
            zip(interactions_df["user_idx"].values,
                interactions_df["item_idx"].values,
                interactions_df["timestamp"].values)
        )

        # Store user histories (only items before current timestamp will be used)
        self.user_histories = user_histories

    def __len__(self):
        return len(self.samples)

    def _get_user_hist_items(self, user_idx, before_ts):
        """Get user's history items before a given timestamp."""
        hist = self.user_histories.get(user_idx, [])
        # Filter items before timestamp and take recent ones
        items = [(item, w) for item, w, ts in hist if ts < before_ts]
        if len(items) > self.max_hist_len:
            items = items[-self.max_hist_len:]
        return items

    def __getitem__(self, idx):
        user_idx, pos_item, ts = self.samples[idx]

        # User history items (before this interaction)
        hist_items = self._get_user_hist_items(user_idx, ts)

        # History item IDs and weights for weighted average
        if hist_items:
            hist_ids = [h[0] for h in hist_items]
            hist_weights = np.array([h[1] for h in hist_items], dtype=np.float32)
            hist_weights = hist_weights / hist_weights.sum()
        else:
            hist_ids = [0]
            hist_weights = np.array([1.0], dtype=np.float32)

        # Positive item category
        pos_cat = self.item_to_cat.get(pos_item, 0)

        # Negative sampling (random items not in user history)
        user_items = set(hist_ids) | {pos_item}
        neg_items = []
        while len(neg_items) < self.n_neg:
            neg = random.randint(0, self.n_items - 1)
            if neg not in user_items:
                neg_items.append(neg)
        neg_cats = [self.item_to_cat.get(n, 0) for n in neg_items]

        return {
            "hist_ids": torch.tensor(hist_ids, dtype=torch.long),
            "hist_weights": torch.tensor(hist_weights, dtype=torch.float32),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "pos_cat": torch.tensor(pos_cat, dtype=torch.long),
            "neg_items": torch.tensor(neg_items, dtype=torch.long),
            "neg_cats": torch.tensor(neg_cats, dtype=torch.long),
        }


def collate_fn(batch):
    """Custom collate: pad history sequences and compute weighted avg later."""
    # Find max history length in batch
    max_len = max(len(b["hist_ids"]) for b in batch)

    hist_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    hist_weights = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, b in enumerate(batch):
        L = len(b["hist_ids"])
        hist_ids[i, :L] = b["hist_ids"]
        hist_weights[i, :L] = b["hist_weights"]

    return {
        "hist_ids": hist_ids,
        "hist_weights": hist_weights,
        "pos_item": torch.stack([b["pos_item"] for b in batch]),
        "pos_cat": torch.stack([b["pos_cat"] for b in batch]),
        "neg_items": torch.stack([b["neg_items"] for b in batch]),
        "neg_cats": torch.stack([b["neg_cats"] for b in batch]),
    }
