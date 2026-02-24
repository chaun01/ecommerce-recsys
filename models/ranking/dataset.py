"""
Dataset and DataLoader for Wide & Deep ranking model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class RankingDataset(Dataset):
    """Dataset for Wide & Deep ranking model."""

    def __init__(
        self,
        interactions_path: str,
        user_embeddings_path: str,
        item_embeddings_path: str,
        item_features_path: str,
        num_negatives: int = 4,
        negative_sampling: str = "random"
    ):
        """
        Initialize ranking dataset.

        Args:
            interactions_path: Path to interactions parquet
            user_embeddings_path: Path to user embeddings
            item_embeddings_path: Path to item embeddings
            item_features_path: Path to item features
            num_negatives: Number of negative samples per positive
            negative_sampling: Negative sampling strategy
        """
        self.num_negatives = num_negatives
        self.negative_sampling = negative_sampling

        # Load data
        print(f"Loading interactions from {interactions_path}...")
        self.interactions = pd.read_parquet(interactions_path)

        print(f"Loading user embeddings from {user_embeddings_path}...")
        self.user_embeddings = np.load(user_embeddings_path)

        print(f"Loading item embeddings from {item_embeddings_path}...")
        self.item_embeddings = np.load(item_embeddings_path)

        print(f"Loading item features from {item_features_path}...")
        self.item_features = pd.read_parquet(item_features_path)

        # Create item feature lookup
        self.item_to_category = dict(
            zip(self.item_features['item_idx'], self.item_features['categoryid'])
        )
        self.item_to_price_bucket = dict(
            zip(self.item_features['item_idx'], self.item_features['price_bucket'])
        )
        self.item_to_price = dict(
            zip(self.item_features['item_idx'], self.item_features['price'])
        )

        # Normalize prices
        prices = self.item_features['price'].values
        self.price_mean = prices.mean()
        self.price_std = prices.std()

        # Get all item IDs for negative sampling
        self.all_items = self.item_features['item_idx'].values
        self.num_items = len(self.all_items)

        # Build user interaction history for negative sampling
        print("Building user interaction sets...")
        self.user_items = {}
        for user_idx, group in self.interactions.groupby('user_idx'):
            self.user_items[user_idx] = set(group['item_idx'].values)

        # Compute item popularity for popularity-based sampling
        if negative_sampling == "popular":
            item_counts = self.interactions['item_idx'].value_counts()
            self.item_popularity = item_counts / item_counts.sum()
            self.popular_items = item_counts.index.values
            self.popular_probs = self.item_popularity.values

        # Compute temporal features
        print("Computing temporal features...")
        self._compute_temporal_features()

        print(f"Dataset initialized with {len(self.interactions):,} interactions")

    def _compute_temporal_features(self):
        """Compute recency and frequency features."""
        # Sort by timestamp
        df = self.interactions.sort_values('timestamp').reset_index(drop=True)

        # For each interaction, compute:
        # 1. Recency: days since last interaction with item
        # 2. Frequency: number of times user interacted before

        self.temporal_features = {}

        for idx, row in df.iterrows():
            user_idx = row['user_idx']
            item_idx = row['item_idx']
            timestamp = row['timestamp']

            # Get user's previous interactions
            prev_interactions = df[
                (df['user_idx'] == user_idx) &
                (df['timestamp'] < timestamp)
            ]

            # Recency: days since last interaction (normalized)
            if len(prev_interactions) > 0:
                last_timestamp = prev_interactions['timestamp'].max()
                recency_ms = timestamp - last_timestamp
                recency_days = recency_ms / (1000 * 60 * 60 * 24)  # Convert to days
                recency_normalized = min(recency_days / 30.0, 1.0)  # Normalize to [0, 1]
            else:
                recency_normalized = 1.0  # First interaction

            # Frequency: normalized count
            frequency = len(prev_interactions)
            frequency_normalized = min(frequency / 100.0, 1.0)  # Normalize to [0, 1]

            self.temporal_features[idx] = (recency_normalized, frequency_normalized)

    def _sample_negatives(self, user_idx: int, positive_item: int) -> List[int]:
        """Sample negative items."""
        negatives = []
        user_items = self.user_items.get(user_idx, set())

        attempts = 0
        max_attempts = self.num_negatives * 10

        while len(negatives) < self.num_negatives and attempts < max_attempts:
            if self.negative_sampling == "random":
                neg_item = np.random.choice(self.all_items)
            elif self.negative_sampling == "popular":
                neg_item = np.random.choice(self.popular_items, p=self.popular_probs)
            else:
                neg_item = np.random.choice(self.all_items)

            if neg_item != positive_item and neg_item not in user_items and neg_item not in negatives:
                negatives.append(neg_item)

            attempts += 1

        # Pad if necessary
        while len(negatives) < self.num_negatives:
            neg_item = np.random.choice(self.all_items)
            if neg_item not in negatives:
                negatives.append(neg_item)

        return negatives

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        row = self.interactions.iloc[idx]

        user_idx = row['user_idx']
        positive_item = row['item_idx']

        # Get user embedding
        user_embedding = self.user_embeddings[user_idx]

        # Get positive item features
        pos_category = self.item_to_category.get(positive_item, 0)
        pos_price_bucket = self.item_to_price_bucket.get(positive_item, -1)
        pos_price = self.item_to_price.get(positive_item, self.price_mean)
        pos_price_normalized = (pos_price - self.price_mean) / (self.price_std + 1e-9)

        # Get temporal features
        recency, frequency = self.temporal_features.get(idx, (1.0, 0.0))

        # Sample negative items
        negative_items = self._sample_negatives(user_idx, positive_item)

        # Get negative item features
        neg_categories = [self.item_to_category.get(item, 0) for item in negative_items]
        neg_price_buckets = [self.item_to_price_bucket.get(item, -1) for item in negative_items]
        neg_prices = [self.item_to_price.get(item, self.price_mean) for item in negative_items]
        neg_prices_normalized = [
            (price - self.price_mean) / (self.price_std + 1e-9)
            for price in neg_prices
        ]

        return {
            # User
            'user_embedding': torch.FloatTensor(user_embedding),
            'user_idx': torch.LongTensor([user_idx]),

            # Positive item
            'pos_item_id': torch.LongTensor([positive_item]),
            'pos_category': torch.LongTensor([pos_category]),
            'pos_price_bucket': torch.LongTensor([pos_price_bucket]),
            'pos_price_normalized': torch.FloatTensor([pos_price_normalized]),

            # Negative items
            'neg_item_ids': torch.LongTensor(negative_items),
            'neg_categories': torch.LongTensor(neg_categories),
            'neg_price_buckets': torch.LongTensor(neg_price_buckets),
            'neg_prices_normalized': torch.FloatTensor(neg_prices_normalized),

            # Temporal features
            'recency': torch.FloatTensor([recency]),
            'frequency': torch.FloatTensor([frequency]),

            # Label (1 for positive, 0 for negatives)
            'label': torch.FloatTensor([1.0])
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    batch_size = len(batch)
    num_negatives = batch[0]['neg_item_ids'].size(0)

    # Stack user embeddings
    user_embeddings = torch.stack([item['user_embedding'] for item in batch])

    # Positive samples
    pos_item_ids = torch.cat([item['pos_item_id'] for item in batch])
    pos_categories = torch.cat([item['pos_category'] for item in batch])
    pos_price_buckets = torch.cat([item['pos_price_bucket'] for item in batch])
    pos_prices_normalized = torch.cat([item['pos_price_normalized'] for item in batch])

    # Negative samples
    neg_item_ids = torch.stack([item['neg_item_ids'] for item in batch])
    neg_categories = torch.stack([item['neg_categories'] for item in batch])
    neg_price_buckets = torch.stack([item['neg_price_buckets'] for item in batch])
    neg_prices_normalized = torch.stack([item['neg_prices_normalized'] for item in batch])

    # Temporal features
    recency = torch.cat([item['recency'] for item in batch])
    frequency = torch.cat([item['frequency'] for item in batch])

    # Combine positive and negative samples
    # Shape: (batch_size * (1 + num_negatives),)
    all_item_ids = torch.cat([
        pos_item_ids,
        neg_item_ids.view(-1)
    ])

    all_categories = torch.cat([
        pos_categories,
        neg_categories.view(-1)
    ])

    all_price_buckets = torch.cat([
        pos_price_buckets,
        neg_price_buckets.view(-1)
    ])

    all_prices_normalized = torch.cat([
        pos_prices_normalized,
        neg_prices_normalized.view(-1)
    ])

    # Repeat user embeddings for each sample (1 positive + num_negatives)
    all_user_embeddings = user_embeddings.repeat_interleave(1 + num_negatives, dim=0)

    # Repeat temporal features
    all_recency = recency.repeat_interleave(1 + num_negatives, dim=0)
    all_frequency = frequency.repeat_interleave(1 + num_negatives, dim=0)

    # Labels: 1 for positive, 0 for negatives
    labels = torch.cat([
        torch.ones(batch_size),
        torch.zeros(batch_size * num_negatives)
    ])

    return {
        'user_embeddings': all_user_embeddings,
        'item_ids': all_item_ids,
        'category_ids': all_categories,
        'price_buckets': all_price_buckets,
        'prices_normalized': all_prices_normalized,
        'recency': all_recency,
        'frequency': all_frequency,
        'labels': labels
    }


def create_ranking_dataloaders(
    train_interactions_path: str,
    val_interactions_path: str,
    user_embeddings_path: str,
    item_embeddings_path: str,
    item_features_path: str,
    batch_size: int = 256,
    num_workers: int = 4,
    num_negatives: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for ranking.

    Args:
        train_interactions_path: Path to train interactions
        val_interactions_path: Path to validation interactions
        user_embeddings_path: Path to user embeddings
        item_embeddings_path: Path to item embeddings
        item_features_path: Path to item features
        batch_size: Batch size
        num_workers: Number of workers
        num_negatives: Number of negative samples

    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("Creating training dataset...")
    train_dataset = RankingDataset(
        interactions_path=train_interactions_path,
        user_embeddings_path=user_embeddings_path,
        item_embeddings_path=item_embeddings_path,
        item_features_path=item_features_path,
        num_negatives=num_negatives
    )

    print("Creating validation dataset...")
    val_dataset = RankingDataset(
        interactions_path=val_interactions_path,
        user_embeddings_path=user_embeddings_path,
        item_embeddings_path=item_embeddings_path,
        item_features_path=item_features_path,
        num_negatives=num_negatives
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader
