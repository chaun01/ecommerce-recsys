"""
Wide & Deep ranking model.
Wide part: feature crosses (user-category, user-item popularity, etc.)
Deep part: concatenated embeddings through MLP.
"""

import torch
import torch.nn as nn


class WideAndDeep(nn.Module):
    def __init__(self, n_items, n_cats, embedding_dim=64, cat_emb_dim=16,
                 wide_dim=32, deep_hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Deep part — item & category embeddings
        self.item_emb = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, cat_emb_dim, padding_idx=0)

        # Deep input: user_emb + item_emb + cat_emb + extra features
        # user_emb (embedding_dim) + item_emb (embedding_dim) + cat_emb (cat_emb_dim) + features (3)
        deep_input_dim = embedding_dim * 2 + cat_emb_dim + 3  # 3 extra: retrieval_score, event_count, recency

        layers = []
        in_dim = deep_input_dim
        for h in deep_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.deep_net = nn.Sequential(*layers)

        # Wide part — linear model on sparse feature crosses
        self.wide_net = nn.Linear(wide_dim, 1, bias=False)

        # Feature cross embedding for wide part
        # Cross: (cat_idx) -> wide_dim
        self.wide_cat_emb = nn.Embedding(n_cats, wide_dim, padding_idx=0)

        # Final prediction
        self.output_layer = nn.Linear(in_dim + 1, 1)  # deep output + wide output

    def forward(self, user_emb, item_ids, cat_ids, retrieval_scores, event_counts, recency):
        """
        Args:
            user_emb: (B, embedding_dim) — user embedding from retrieval model
            item_ids: (B,) — candidate item indices
            cat_ids: (B,) — candidate item category indices
            retrieval_scores: (B,) — FAISS retrieval scores
            event_counts: (B,) — number of times user interacted with this category
            recency: (B,) — normalized recency feature
        Returns:
            scores: (B,) relevance scores
        """
        # Deep part
        ie = self.item_emb(item_ids)                      # (B, D)
        ce = self.cat_emb(cat_ids)                        # (B, cat_emb_dim)
        extra = torch.stack([retrieval_scores, event_counts, recency], dim=-1)  # (B, 3)

        deep_input = torch.cat([user_emb, ie, ce, extra], dim=-1)
        deep_out = self.deep_net(deep_input)              # (B, last_hidden)

        # Wide part
        wide_feat = self.wide_cat_emb(cat_ids)            # (B, wide_dim)
        wide_out = self.wide_net(wide_feat)               # (B, 1)

        # Combine
        combined = torch.cat([deep_out, wide_out], dim=-1)
        score = self.output_layer(combined).squeeze(-1)   # (B,)
        return score
