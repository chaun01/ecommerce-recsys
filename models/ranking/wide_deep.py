"""
Wide & Deep Model for Ranking.

Combines memorization (Wide) and generalization (Deep) for accurate ranking.

Reference:
- Wide & Deep Learning for Recommender Systems (Cheng et al., 2016)
  https://arxiv.org/abs/1606.07792
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class WideComponent(nn.Module):
    """
    Wide component for memorization through feature crosses.

    Learns direct feature interactions via linear model.
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        cross_features: List[Tuple[str, str]]
    ):
        """
        Initialize Wide component.

        Args:
            feature_dims: Dictionary of feature names to dimensions
            cross_features: List of feature pairs to cross
        """
        super().__init__()

        self.feature_dims = feature_dims
        self.cross_features = cross_features

        # Create embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        for feat_name, feat_dim in feature_dims.items():
            self.embeddings[feat_name] = nn.Embedding(feat_dim, 1)

        # Calculate total dimension for crossed features
        self.total_dim = 0
        for feat1, feat2 in cross_features:
            dim1 = feature_dims[feat1]
            dim2 = feature_dims[feat2]
            self.total_dim += dim1 * dim2

        # Linear layer for crossed features
        if self.total_dim > 0:
            self.cross_linear = nn.Linear(self.total_dim, 1, bias=False)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Dictionary of feature tensors

        Returns:
            Wide output of shape (batch_size, 1)
        """
        batch_size = list(features.values())[0].size(0)

        # Get individual feature contributions
        wide_out = torch.zeros(batch_size, 1, device=list(features.values())[0].device)

        for feat_name, feat_tensor in features.items():
            if feat_name in self.embeddings:
                wide_out += self.embeddings[feat_name](feat_tensor).squeeze(-1)

        # Add crossed feature contributions
        if self.total_dim > 0:
            crossed = []
            for feat1, feat2 in self.cross_features:
                if feat1 in features and feat2 in features:
                    # Get embeddings for both features
                    emb1 = F.one_hot(features[feat1], self.feature_dims[feat1]).float()
                    emb2 = F.one_hot(features[feat2], self.feature_dims[feat2]).float()

                    # Outer product to create crossed features
                    cross = torch.bmm(
                        emb1.unsqueeze(-1),  # (batch, dim1, 1)
                        emb2.unsqueeze(1)    # (batch, 1, dim2)
                    )  # (batch, dim1, dim2)

                    crossed.append(cross.view(batch_size, -1))

            if crossed:
                crossed_features = torch.cat(crossed, dim=1)
                wide_out += self.cross_linear(crossed_features)

        return wide_out


class DeepComponent(nn.Module):
    """
    Deep component for generalization through neural network.

    Learns complex non-linear feature interactions via MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize Deep component.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
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

        # Build MLP layers
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
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Deep output of shape (batch_size, 1)
        """
        return self.network(x)


class WideAndDeepModel(nn.Module):
    """
    Wide & Deep model combining memorization and generalization.
    """

    def __init__(
        self,
        # Embedding dimensions
        num_items: int,
        num_categories: int,
        user_embedding_dim: int = 64,
        item_embedding_dim: int = 64,

        # Wide component
        wide_feature_dims: Optional[Dict[str, int]] = None,
        cross_features: Optional[List[Tuple[str, str]]] = None,

        # Deep component
        deep_hidden_dims: List[int] = [128, 64, 32],
        activation: str = "relu",
        dropout: float = 0.3,
        use_batch_norm: bool = True,

        # Additional features
        use_price_features: bool = True,
        use_temporal_features: bool = True
    ):
        """
        Initialize Wide & Deep model.

        Args:
            num_items: Number of items
            num_categories: Number of categories
            user_embedding_dim: User embedding dimension
            item_embedding_dim: Item embedding dimension
            wide_feature_dims: Feature dimensions for wide component
            cross_features: Feature pairs to cross in wide component
            deep_hidden_dims: Hidden dimensions for deep component
            activation: Activation function
            dropout: Dropout probability
            use_batch_norm: Use batch normalization
            use_price_features: Use price-related features
            use_temporal_features: Use temporal features
        """
        super().__init__()

        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.use_price_features = use_price_features
        self.use_temporal_features = use_temporal_features

        # Item and category embeddings for deep component
        self.item_embedding = nn.Embedding(num_items, item_embedding_dim)
        self.category_embedding = nn.Embedding(num_categories + 1, item_embedding_dim // 2)

        # Wide component
        if wide_feature_dims and cross_features:
            self.use_wide = True
            self.wide = WideComponent(wide_feature_dims, cross_features)
        else:
            self.use_wide = False

        # Calculate deep input dimension
        deep_input_dim = user_embedding_dim + item_embedding_dim + item_embedding_dim // 2

        if use_price_features:
            deep_input_dim += 2  # price_bucket, price_normalized

        if use_temporal_features:
            deep_input_dim += 2  # recency, frequency

        # Deep component
        self.deep = DeepComponent(
            input_dim=deep_input_dim,
            hidden_dims=deep_hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
        price_features: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        wide_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            user_embeddings: User embeddings (batch_size, user_embedding_dim)
            item_ids: Item IDs (batch_size,)
            category_ids: Category IDs (batch_size,)
            price_features: Price features (batch_size, 2) [bucket, normalized]
            temporal_features: Temporal features (batch_size, 2) [recency, frequency]
            wide_features: Dictionary of categorical features for wide component

        Returns:
            Predicted scores (batch_size, 1)
        """
        # Get item embeddings
        item_embeds = self.item_embedding(item_ids)  # (batch_size, item_embedding_dim)
        cat_embeds = self.category_embedding(category_ids)  # (batch_size, item_embedding_dim // 2)

        # Build deep input
        deep_features = [user_embeddings, item_embeds, cat_embeds]

        if self.use_price_features and price_features is not None:
            deep_features.append(price_features)

        if self.use_temporal_features and temporal_features is not None:
            deep_features.append(temporal_features)

        deep_input = torch.cat(deep_features, dim=-1)

        # Deep component
        deep_out = self.deep(deep_input)

        # Wide component
        if self.use_wide and wide_features is not None:
            wide_out = self.wide(wide_features)
            output = wide_out + deep_out
        else:
            output = deep_out

        return output

    def predict(
        self,
        user_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
        price_features: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        wide_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Predict scores with sigmoid activation.

        Returns:
            Predicted probabilities (batch_size,)
        """
        logits = self.forward(
            user_embeddings, item_ids, category_ids,
            price_features, temporal_features, wide_features
        )
        return torch.sigmoid(logits.squeeze(-1))


def create_wide_deep_model(config: Dict, num_items: int, num_categories: int) -> WideAndDeepModel:
    """
    Create Wide & Deep model from config.

    Args:
        config: Configuration dictionary
        num_items: Number of items
        num_categories: Number of categories

    Returns:
        WideAndDeepModel instance
    """
    model_config = config["model"]
    deep_config = model_config.get("deep", {})

    # Wide feature dimensions (if using wide component)
    wide_config = model_config.get("wide", {})
    cross_features = wide_config.get("cross_features", [])

    if cross_features:
        # Define feature dimensions for wide component
        wide_feature_dims = {
            "user_category_hist": num_categories,
            "item_category": num_categories,
            "price_bucket": 6,  # 0-4 + unknown
            "user_price_pref": 6,
            "recency_bucket": 6,
            "item_popularity": 10
        }

        # Convert cross features from list of lists to list of tuples
        cross_features_tuples = [tuple(pair) for pair in cross_features]
    else:
        wide_feature_dims = None
        cross_features_tuples = None

    return WideAndDeepModel(
        num_items=num_items,
        num_categories=num_categories,
        user_embedding_dim=config.get("retrieval_embedding_dim", 64),
        item_embedding_dim=deep_config.get("item_embedding_dim", 64),
        wide_feature_dims=wide_feature_dims,
        cross_features=cross_features_tuples,
        deep_hidden_dims=deep_config.get("hidden_dims", [128, 64, 32]),
        activation=deep_config.get("activation", "relu"),
        dropout=deep_config.get("dropout", 0.3),
        use_batch_norm=deep_config.get("batch_norm", True),
        use_price_features=model_config.get("use_price_features", True),
        use_temporal_features=model_config.get("use_temporal_features", True)
    )
