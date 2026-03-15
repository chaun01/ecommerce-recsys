"""
Two-Tower Neural Retrieval Model.
User tower: aggregates recent interaction history (item embeddings + event weights).
Item tower: item embedding + category embedding.
"""

import torch
import torch.nn as nn


class UserTower(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, user_hist_emb):
        """
        Args:
            user_hist_emb: (batch, embedding_dim) — weighted average of item embeddings
        Returns:
            (batch, embedding_dim) user representation
        """
        return self.fc(user_hist_emb)


class ItemTower(nn.Module):
    def __init__(self, n_items, n_cats, embedding_dim, cat_embedding_dim=16):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, cat_embedding_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + cat_embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, item_ids, cat_ids):
        """
        Args:
            item_ids: (batch,) item indices
            cat_ids: (batch,) category indices
        Returns:
            (batch, embedding_dim)
        """
        ie = self.item_emb(item_ids)
        ce = self.cat_emb(cat_ids)
        return self.fc(torch.cat([ie, ce], dim=-1))


class TwoTowerModel(nn.Module):
    def __init__(self, n_items, n_cats, embedding_dim=64, hidden_dim=128, cat_embedding_dim=16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.item_tower = ItemTower(n_items, n_cats, embedding_dim, cat_embedding_dim)
        self.user_tower = UserTower(embedding_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def get_item_embeddings(self, item_ids, cat_ids):
        return self.item_tower(item_ids, cat_ids)

    def get_user_embeddings(self, user_hist_emb):
        return self.user_tower(user_hist_emb)

    def forward(self, user_hist_emb, pos_item_ids, pos_cat_ids, neg_item_ids, neg_cat_ids):
        """
        Compute contrastive loss (in-batch negatives + explicit negatives).
        Args:
            user_hist_emb: (batch, emb_dim)
            pos_item_ids: (batch,)
            pos_cat_ids: (batch,)
            neg_item_ids: (batch, n_neg)
            neg_cat_ids: (batch, n_neg)
        """
        user_emb = self.user_tower(user_hist_emb)                     # (B, D)
        pos_emb = self.item_tower(pos_item_ids, pos_cat_ids)          # (B, D)

        B, n_neg = neg_item_ids.shape
        neg_emb = self.item_tower(
            neg_item_ids.reshape(-1), neg_cat_ids.reshape(-1)
        ).reshape(B, n_neg, -1)                                       # (B, n_neg, D)

        # Normalize
        user_emb = nn.functional.normalize(user_emb, dim=-1)
        pos_emb = nn.functional.normalize(pos_emb, dim=-1)
        neg_emb = nn.functional.normalize(neg_emb, dim=-1)

        # Scores
        pos_score = (user_emb * pos_emb).sum(dim=-1, keepdim=True)    # (B, 1)
        neg_score = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1)  # (B, n_neg)

        logits = torch.cat([pos_score, neg_score], dim=-1) / self.temperature.abs()
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)  # pos is index 0

        loss = nn.functional.cross_entropy(logits, labels)
        return loss
