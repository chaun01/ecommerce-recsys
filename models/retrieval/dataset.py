"""
Dataset and DataLoader for Two-Tower retrieval model training.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import random


class RetrievalDataset(Dataset):
    """Dataset for Two-Tower retrieval model."""

    def __init__(
        self,
        interactions_path: str,
        sequences_path: str,
        item_features_path: str,
        max_sequence_length: int = 50,
        num_negatives: int = 4,
        negative_sampling: str = "random"
    ):
        """
        Initialize dataset.

        Args:
            interactions_path: Path to interactions parquet file
            sequences_path: Path to user sequences parquet file
            item_features_path: Path to item features parquet file
            max_sequence_length: Maximum sequence length
            num_negatives: Number of negative samples per positive
            negative_sampling: Negative sampling strategy ('random', 'popular')
        """
        self.max_sequence_length = max_sequence_length
        self.num_negatives = num_negatives
        self.negative_sampling = negative_sampling

        # Load data
        print(f"Loading interactions from {interactions_path}...")
        self.interactions = pd.read_parquet(interactions_path)

        print(f"Loading sequences from {sequences_path}...")
        self.sequences = pd.read_parquet(sequences_path)

        print(f"Loading item features from {item_features_path}...")
        self.item_features = pd.read_parquet(item_features_path)

        # Create item feature lookup
        self.item_to_category = dict(
            zip(self.item_features['item_idx'], self.item_features['categoryid'])
        )

        # Get all item IDs for negative sampling
        self.all_items = self.item_features['item_idx'].values
        self.num_items = len(self.all_items)

        # Build user interaction history
        print("Building user interaction history...")
        self._build_user_history()

        # Compute item popularity for popularity-based negative sampling
        if negative_sampling == "popular":
            item_counts = self.interactions['item_idx'].value_counts()
            self.item_popularity = item_counts / item_counts.sum()
            self.popular_items = item_counts.index.values
            self.popular_probs = self.item_popularity.values

        print(f"Dataset initialized with {len(self)} samples")

    def _build_user_history(self):
        """Build user interaction history excluding target items."""
        # Sort interactions by timestamp
        self.interactions = self.interactions.sort_values('timestamp').reset_index(drop=True)

        # Group by user
        user_groups = self.interactions.groupby('user_idx')

        # For each interaction, get previous items
        self.samples = []

        for user_idx, group in user_groups:
            items = group['item_idx'].values
            timestamps = group['timestamp'].values

            # For each item (except first), create a sample
            for i in range(1, len(items)):
                target_item = items[i]
                target_timestamp = timestamps[i]

                # Get previous items (up to max_sequence_length)
                history = items[:i][-self.max_sequence_length:]

                self.samples.append({
                    'user_idx': user_idx,
                    'history': history,
                    'target_item': target_item,
                    'timestamp': target_timestamp
                })

    def _sample_negatives(self, user_idx: int, positive_item: int) -> List[int]:
        """Sample negative items for a user."""
        negatives = []

        # Get user's interacted items to avoid sampling them
        user_items = set(self.interactions[self.interactions['user_idx'] == user_idx]['item_idx'].values)

        attempts = 0
        max_attempts = self.num_negatives * 10

        while len(negatives) < self.num_negatives and attempts < max_attempts:
            if self.negative_sampling == "random":
                # Uniform random sampling
                neg_item = np.random.choice(self.all_items)
            elif self.negative_sampling == "popular":
                # Popularity-based sampling
                neg_item = np.random.choice(self.popular_items, p=self.popular_probs)
            else:
                neg_item = np.random.choice(self.all_items)

            # Ensure negative is not positive or already interacted
            if neg_item != positive_item and neg_item not in user_items and neg_item not in negatives:
                negatives.append(neg_item)

            attempts += 1

        # If we couldn't sample enough negatives, pad with random items
        while len(negatives) < self.num_negatives:
            neg_item = np.random.choice(self.all_items)
            if neg_item not in negatives:
                negatives.append(neg_item)

        return negatives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        sample = self.samples[idx]

        user_idx = sample['user_idx']
        history = sample['history']
        target_item = sample['target_item']

        # Pad or truncate history
        if len(history) < self.max_sequence_length:
            # Pad with zeros
            padded_history = np.pad(
                history,
                (0, self.max_sequence_length - len(history)),
                mode='constant',
                constant_values=0
            )
            sequence_length = len(history)
        else:
            padded_history = history[-self.max_sequence_length:]
            sequence_length = self.max_sequence_length

        # Sample negative items
        negative_items = self._sample_negatives(user_idx, target_item)

        # Get categories
        target_category = self.item_to_category.get(target_item, 0)
        negative_categories = [self.item_to_category.get(item, 0) for item in negative_items]

        return {
            'user_item_sequence': torch.LongTensor(padded_history),
            'sequence_length': torch.LongTensor([sequence_length]),
            'positive_item': torch.LongTensor([target_item]),
            'positive_category': torch.LongTensor([target_category]),
            'negative_items': torch.LongTensor(negative_items),
            'negative_categories': torch.LongTensor(negative_categories),
            'user_idx': torch.LongTensor([user_idx])
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        'user_item_sequence': torch.stack([item['user_item_sequence'] for item in batch]),
        'sequence_length': torch.cat([item['sequence_length'] for item in batch]),
        'positive_items': torch.cat([item['positive_item'] for item in batch]),
        'positive_categories': torch.cat([item['positive_category'] for item in batch]),
        'negative_items': torch.stack([item['negative_items'] for item in batch]),
        'negative_categories': torch.stack([item['negative_categories'] for item in batch]),
        'user_idx': torch.cat([item['user_idx'] for item in batch])
    }


def create_dataloaders(
    train_interactions_path: str,
    val_interactions_path: str,
    train_sequences_path: str,
    item_features_path: str,
    batch_size: int = 512,
    num_workers: int = 4,
    max_sequence_length: int = 50,
    num_negatives: int = 4,
    negative_sampling: str = "random"
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_interactions_path: Path to train interactions
        val_interactions_path: Path to validation interactions
        train_sequences_path: Path to train sequences
        item_features_path: Path to item features
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_sequence_length: Maximum sequence length
        num_negatives: Number of negative samples
        negative_sampling: Negative sampling strategy

    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("Creating training dataset...")
    train_dataset = RetrievalDataset(
        interactions_path=train_interactions_path,
        sequences_path=train_sequences_path,
        item_features_path=item_features_path,
        max_sequence_length=max_sequence_length,
        num_negatives=num_negatives,
        negative_sampling=negative_sampling
    )

    print("Creating validation dataset...")
    val_dataset = RetrievalDataset(
        interactions_path=val_interactions_path,
        sequences_path=train_sequences_path,  # Use train sequences for validation
        item_features_path=item_features_path,
        max_sequence_length=max_sequence_length,
        num_negatives=num_negatives,
        negative_sampling=negative_sampling
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
