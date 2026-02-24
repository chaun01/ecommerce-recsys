"""
Two-Tower Neural Network for Retrieval.

This model learns separate embeddings for users and items that can be used
for fast candidate generation via approximate nearest neighbor search.

Architecture:
- User Tower: Encodes user interaction history into dense embedding
- Item Tower: Encodes item features into dense embedding
- Training: Contrastive learning (InfoNCE loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MLP(nn.Module):
    """Multi-Layer Perceptron with batch normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize MLP.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class UserTower(nn.Module):
    """User tower that encodes user interaction history."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.2,
        pooling: str = "mean"
    ):
        """
        Initialize User Tower.

        Args:
            num_items: Number of items in catalog
            embedding_dim: Item embedding dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Final user embedding dimension
            activation: Activation function
            dropout: Dropout probability
            pooling: Pooling method ('mean', 'max', 'sum')
        """
        super().__init__()

        self.pooling = pooling

        # Item embeddings (shared with item tower)
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=0
        )

        # MLP to process pooled embeddings
        self.mlp = MLP(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout
        )

    def forward(
        self,
        item_sequence: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode user from item sequence.

        Args:
            item_sequence: Tensor of shape (batch_size, seq_len)
            sequence_lengths: Actual sequence lengths (batch_size,)

        Returns:
            User embeddings of shape (batch_size, output_dim)
        """
        # Get item embeddings: (batch_size, seq_len, embedding_dim)
        item_embeds = self.item_embedding(item_sequence)

        # Pool embeddings
        if self.pooling == "mean":
            if sequence_lengths is not None:
                # Masked mean pooling - fix CUDA error
                batch_size, seq_len, _ = item_embeds.size()
                # Clamp sequence lengths to valid range [0, seq_len]
                sequence_lengths = torch.clamp(sequence_lengths, 0, seq_len)
                # Create indices on CPU, then move to device
                indices = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
                indices = indices.to(sequence_lengths.device)
                # Compare on same device
                mask = (indices < sequence_lengths.unsqueeze(1)).unsqueeze(-1).float()
                pooled = (item_embeds * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            else:
                pooled = item_embeds.mean(dim=1)

        elif self.pooling == "max":
            pooled = item_embeds.max(dim=1)[0]

        elif self.pooling == "sum":
            pooled = item_embeds.sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Pass through MLP
        user_embed = self.mlp(pooled)

        # L2 normalize
        user_embed = F.normalize(user_embed, p=2, dim=1)

        return user_embed


class ItemTower(nn.Module):
    """Item tower that encodes item features."""

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        embedding_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.2,
        use_category: bool = True
    ):
        """
        Initialize Item Tower.

        Args:
            num_items: Number of items in catalog
            num_categories: Number of categories
            embedding_dim: Embedding dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Final item embedding dimension
            activation: Activation function
            dropout: Dropout probability
            use_category: Whether to use category features
        """
        super().__init__()

        self.use_category = use_category

        # Item ID embedding
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=0
        )

        # Category embedding
        if use_category:
            self.category_embedding = nn.Embedding(
                num_categories + 1,  # +1 for unknown category
                embedding_dim // 2
            )
            mlp_input_dim = embedding_dim + embedding_dim // 2
        else:
            mlp_input_dim = embedding_dim

        # MLP to process features
        self.mlp = MLP(
            input_dim=mlp_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode items from features.

        Args:
            item_ids: Item IDs of shape (batch_size,)
            category_ids: Category IDs of shape (batch_size,)

        Returns:
            Item embeddings of shape (batch_size, output_dim)
        """
        # Get item embeddings
        item_embeds = self.item_embedding(item_ids)

        # Concatenate category embeddings if available
        if self.use_category and category_ids is not None:
            cat_embeds = self.category_embedding(category_ids)
            features = torch.cat([item_embeds, cat_embeds], dim=-1)
        else:
            features = item_embeds

        # Pass through MLP
        item_embed = self.mlp(features)

        # L2 normalize
        item_embed = F.normalize(item_embed, p=2, dim=1)

        return item_embed


class TwoTowerModel(nn.Module):
    """Two-Tower model for retrieval."""

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        embedding_dim: int = 64,
        user_hidden_dims: List[int] = [256, 128],
        item_hidden_dims: List[int] = [256, 128],
        output_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.2,
        user_pooling: str = "mean",
        use_category: bool = True,
        temperature: float = 0.07
    ):
        """
        Initialize Two-Tower Model.

        Args:
            num_items: Number of items in catalog
            num_categories: Number of categories
            embedding_dim: Embedding dimension
            user_hidden_dims: User tower hidden dimensions
            item_hidden_dims: Item tower hidden dimensions
            output_dim: Final embedding dimension (user and item)
            activation: Activation function
            dropout: Dropout probability
            user_pooling: User pooling method
            use_category: Whether to use category features
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.temperature = temperature

        # User tower
        self.user_tower = UserTower(
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dims=user_hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            pooling=user_pooling
        )

        # Item tower
        self.item_tower = ItemTower(
            num_items=num_items,
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            hidden_dims=item_hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            use_category=use_category
        )

    def forward(
        self,
        user_item_sequence: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: Optional[torch.Tensor] = None,
        user_sequence_lengths: Optional[torch.Tensor] = None,
        positive_categories: Optional[torch.Tensor] = None,
        negative_categories: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            user_item_sequence: User's item history (batch_size, seq_len)
            positive_items: Positive item IDs (batch_size,)
            negative_items: Negative item IDs (batch_size, num_negatives)
            user_sequence_lengths: Actual sequence lengths
            positive_categories: Positive item categories
            negative_categories: Negative item categories

        Returns:
            Dictionary with embeddings and loss
        """
        batch_size = user_item_sequence.size(0)

        # Encode users
        user_embeds = self.user_tower(
            user_item_sequence,
            user_sequence_lengths
        )  # (batch_size, output_dim)

        # Encode positive items
        pos_item_embeds = self.item_tower(
            positive_items,
            positive_categories
        )  # (batch_size, output_dim)

        # Compute similarity scores
        # Positive scores: (batch_size,)
        pos_scores = (user_embeds * pos_item_embeds).sum(dim=1) / self.temperature

        # In-batch negatives: use all other items in batch as negatives
        # Similarity matrix: (batch_size, batch_size)
        all_scores = torch.matmul(user_embeds, pos_item_embeds.T) / self.temperature

        # Compute InfoNCE loss (contrastive loss)
        labels = torch.arange(batch_size, device=user_embeds.device)
        loss = F.cross_entropy(all_scores, labels)

        # If explicit negatives are provided, add them
        if negative_items is not None:
            num_negatives = negative_items.size(1)
            # Flatten negatives: (batch_size * num_negatives,)
            neg_items_flat = negative_items.view(-1)
            neg_cats_flat = negative_categories.view(-1) if negative_categories is not None else None

            # Encode negatives: (batch_size * num_negatives, output_dim)
            neg_item_embeds = self.item_tower(neg_items_flat, neg_cats_flat)

            # Reshape: (batch_size, num_negatives, output_dim)
            neg_item_embeds = neg_item_embeds.view(batch_size, num_negatives, -1)

            # Compute negative scores: (batch_size, num_negatives)
            neg_scores = torch.bmm(
                user_embeds.unsqueeze(1),  # (batch_size, 1, output_dim)
                neg_item_embeds.transpose(1, 2)  # (batch_size, output_dim, num_negatives)
            ).squeeze(1) / self.temperature  # (batch_size, num_negatives)

            # Concatenate positive and negative scores
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

            # Labels: positive is always index 0
            labels = torch.zeros(batch_size, dtype=torch.long, device=user_embeds.device)
            loss = F.cross_entropy(all_scores, labels)

        return {
            "loss": loss,
            "user_embeds": user_embeds,
            "item_embeds": pos_item_embeds,
            "pos_scores": pos_scores
        }

    def encode_user(
        self,
        user_item_sequence: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode user from item sequence."""
        return self.user_tower(user_item_sequence, sequence_lengths)

    def encode_item(
        self,
        item_ids: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode items from features."""
        return self.item_tower(item_ids, category_ids)

    def predict_score(
        self,
        user_embeds: torch.Tensor,
        item_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict scores between users and items.

        Args:
            user_embeds: User embeddings (batch_size, output_dim)
            item_embeds: Item embeddings (num_items, output_dim)

        Returns:
            Scores of shape (batch_size, num_items)
        """
        # Dot product similarity
        scores = torch.matmul(user_embeds, item_embeds.T)
        return scores


def create_two_tower_model(config: Dict) -> TwoTowerModel:
    """
    Create Two-Tower model from config.

    Args:
        config: Configuration dictionary

    Returns:
        TwoTowerModel instance
    """
    model_config = config["model"]

    return TwoTowerModel(
        num_items=config["data"]["num_items"],
        num_categories=config["data"]["num_categories"],
        embedding_dim=model_config.get("embedding_dim", 64),
        user_hidden_dims=model_config.get("user_tower", {}).get("hidden_dims", [256, 128, 64]),
        item_hidden_dims=model_config.get("item_tower", {}).get("hidden_dims", [256, 128, 64]),
        output_dim=model_config.get("embedding_dim", 64),
        activation=model_config.get("user_tower", {}).get("activation", "relu"),
        dropout=model_config.get("user_tower", {}).get("dropout", 0.2),
        user_pooling=model_config.get("user_pooling", "mean"),
        use_category=model_config.get("use_category", True),
        temperature=config["training"].get("temperature", 0.07)
    )
