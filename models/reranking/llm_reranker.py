"""
LLM-based Reranker for semantic understanding.

Uses a pre-trained language model to rerank candidates based on
semantic relevance between user history and candidate items.

Approaches:
1. Pointwise: Score each candidate independently
2. Pairwise: Compare pairs of candidates
3. Listwise: Rerank entire list at once
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from transformers import AutoTokenizer, AutoModel


class LLMReranker(nn.Module):
    """
    LLM-based reranker using pre-trained transformer.

    This is a lightweight implementation using sentence transformers
    for semantic similarity between user history and candidate items.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize LLM reranker.

        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            device: Device to run model on
        """
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.device = device

        # Load tokenizer and model
        print(f"Loading LLM reranker: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.to(device)

        # Scoring head
        hidden_size = self.encoder.config.hidden_size
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_size * 3, 256),  # [user, item, user*item]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(device)

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over token embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings

        Returns:
            Embeddings of shape (len(texts), hidden_size)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.encoder(**encoded)
            embeddings = self.mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])

        return embeddings

    def forward(
        self,
        user_texts: List[str],
        item_texts: List[str]
    ) -> torch.Tensor:
        """
        Score user-item pairs.

        Args:
            user_texts: User history descriptions
            item_texts: Candidate item descriptions

        Returns:
            Scores of shape (batch_size,)
        """
        # Encode texts
        user_embeds = self.encode_text(user_texts)  # (batch_size, hidden_size)
        item_embeds = self.encode_text(item_texts)  # (batch_size, hidden_size)

        # Create interaction features
        interaction = user_embeds * item_embeds  # Element-wise product

        # Concatenate features
        features = torch.cat([user_embeds, item_embeds, interaction], dim=-1)

        # Score
        scores = self.scoring_head(features).squeeze(-1)

        return scores

    def rerank(
        self,
        user_history: List[str],
        candidates: List[str],
        candidate_scores: Optional[List[float]] = None,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> Tuple[List[int], List[float]]:
        """
        Rerank candidates based on semantic relevance.

        Args:
            user_history: List of item descriptions in user history
            candidates: List of candidate item descriptions
            candidate_scores: Optional scores from previous ranker
            top_k: Number of items to return
            alpha: Weight for combining scores (alpha * llm + (1-alpha) * previous)

        Returns:
            Tuple of (indices, scores) for top-k items
        """
        # Create user context (concatenate recent history)
        user_context = " | ".join(user_history[-5:])  # Last 5 items
        user_texts = [user_context] * len(candidates)

        # Get LLM scores
        with torch.no_grad():
            llm_scores = self.forward(user_texts, candidates)
            llm_scores = torch.sigmoid(llm_scores).cpu().numpy()

        # Combine with previous scores if provided
        if candidate_scores is not None:
            final_scores = alpha * llm_scores + (1 - alpha) * candidate_scores
        else:
            final_scores = llm_scores

        # Get top-k
        top_indices = final_scores.argsort()[-top_k:][::-1]
        top_scores = final_scores[top_indices]

        return top_indices.tolist(), top_scores.tolist()


class SimpleLLMReranker:
    """
    Simplified LLM reranker using cosine similarity.

    No training required - uses pre-trained embeddings directly.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize simple reranker."""
        print(f"Loading simple LLM reranker: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.to(device)
        self.encoder.eval()
        self.device = device

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.encoder(**encoded)
            embeddings = self.mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def rerank(
        self,
        user_history: List[str],
        candidates: List[str],
        candidate_scores: Optional[List[float]] = None,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> Tuple[List[int], List[float]]:
        """
        Rerank using cosine similarity.

        Args:
            user_history: User's item history as text
            candidates: Candidate items as text
            candidate_scores: Previous scores
            top_k: Number to return
            alpha: Combination weight

        Returns:
            Top-k indices and scores
        """
        # Encode user profile (average of history)
        if len(user_history) > 0:
            user_embeds = self.encode(user_history[-5:])  # Last 5 items
            user_profile = user_embeds.mean(dim=0, keepdim=True)  # Average
        else:
            # Empty history - return based on previous scores
            if candidate_scores is not None:
                top_indices = sorted(range(len(candidate_scores)),
                                   key=lambda i: candidate_scores[i],
                                   reverse=True)[:top_k]
                top_scores = [candidate_scores[i] for i in top_indices]
                return top_indices, top_scores
            else:
                return list(range(min(top_k, len(candidates)))), [0.0] * min(top_k, len(candidates))

        # Encode candidates
        candidate_embeds = self.encode(candidates)

        # Compute cosine similarity
        similarities = torch.mm(user_profile, candidate_embeds.T).squeeze(0)
        llm_scores = similarities.cpu().numpy()

        # Normalize to [0, 1]
        llm_scores = (llm_scores + 1) / 2

        # Combine scores
        if candidate_scores is not None:
            import numpy as np
            candidate_scores = np.array(candidate_scores)
            final_scores = alpha * llm_scores + (1 - alpha) * candidate_scores
        else:
            final_scores = llm_scores

        # Get top-k
        top_indices = final_scores.argsort()[-top_k:][::-1]
        top_scores = final_scores[top_indices]

        return top_indices.tolist(), top_scores.tolist()


def format_item_text(item_data: Dict) -> str:
    """
    Format item data into text for LLM.

    Args:
        item_data: Dictionary with item information

    Returns:
        Formatted text description
    """
    parts = []

    if 'title' in item_data:
        parts.append(item_data['title'])

    if 'category' in item_data:
        parts.append(f"Category: {item_data['category']}")

    if 'price' in item_data:
        parts.append(f"Price: ${item_data['price']}")

    if 'description' in item_data:
        parts.append(item_data['description'][:200])  # Truncate

    return " | ".join(parts) if parts else "Unknown item"


def create_llm_reranker(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    simple: bool = True
) -> SimpleLLMReranker:
    """
    Create LLM reranker.

    Args:
        model_name: HuggingFace model name
        simple: Use simple version (no training)

    Returns:
        Reranker instance
    """
    if simple:
        return SimpleLLMReranker(model_name)
    else:
        return LLMReranker(model_name)
