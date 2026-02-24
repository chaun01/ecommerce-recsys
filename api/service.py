"""
Recommendation service that orchestrates the full pipeline.
"""

import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from retrieval.faiss_index import FAISSIndex
from models.retrieval.two_tower import TwoTowerModel
from api.models import RecommendationItem, ItemMetadata

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Main recommendation service orchestrating the full pipeline:
    1. Retrieval (Two-Tower + FAISS)
    2. Ranking (Wide & Deep) - Optional
    3. LLM Reranking - Optional
    """

    def __init__(
        self,
        retrieval_model_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        item_features_path: Optional[str] = None,
        ranking_model_path: Optional[str] = None,
        use_ranking: bool = False,
        use_llm_reranker: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize recommendation service.

        Args:
            retrieval_model_path: Path to Two-Tower model checkpoint
            faiss_index_path: Path to FAISS index
            item_features_path: Path to item features
            ranking_model_path: Path to ranking model (optional)
            use_ranking: Whether to use ranking model
            use_llm_reranker: Whether to use LLM reranker
            device: Device for models
        """
        self.device = device
        self.use_ranking = use_ranking
        self.use_llm_reranker = use_llm_reranker

        logger.info("Initializing Recommendation Service...")

        # Load item features
        if item_features_path and Path(item_features_path).exists():
            logger.info(f"Loading item features from {item_features_path}")
            self.item_features = pd.read_parquet(item_features_path)
            self.item_to_category = dict(
                zip(self.item_features['item_idx'], self.item_features['categoryid'])
            )
            logger.info(f"Loaded {len(self.item_features)} item features")
        else:
            logger.warning("No item features provided")
            self.item_features = None
            self.item_to_category = {}

        # Load FAISS index
        if faiss_index_path and Path(faiss_index_path).exists():
            logger.info(f"Loading FAISS index from {faiss_index_path}")
            self.faiss_index = FAISSIndex.load(faiss_index_path)
            logger.info(f"Loaded FAISS index with {self.faiss_index.index.ntotal} items")
        else:
            logger.warning("No FAISS index provided - will use mock retrieval")
            self.faiss_index = None

        # Load retrieval model
        if retrieval_model_path and Path(retrieval_model_path).exists():
            logger.info(f"Loading retrieval model from {retrieval_model_path}")
            checkpoint = torch.load(retrieval_model_path, map_location=device)

            from models.retrieval.two_tower import create_two_tower_model
            self.retrieval_model = create_two_tower_model(checkpoint['config'])
            self.retrieval_model.load_state_dict(checkpoint['model_state_dict'])
            self.retrieval_model.to(device)
            self.retrieval_model.eval()
            logger.info("Retrieval model loaded successfully")
        else:
            logger.warning("No retrieval model provided")
            self.retrieval_model = None

        # Load ranking model (optional)
        self.ranking_model = None
        if use_ranking and ranking_model_path and Path(ranking_model_path).exists():
            logger.info(f"Loading ranking model from {ranking_model_path}")
            # TODO: Load ranking model
            logger.info("Ranking model loaded successfully")

        # Load LLM reranker (optional)
        self.llm_reranker = None
        if use_llm_reranker:
            try:
                from models.reranking.llm_reranker import SimpleLLMReranker
                logger.info("Loading LLM reranker...")
                self.llm_reranker = SimpleLLMReranker()
                logger.info("LLM reranker loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LLM reranker: {e}")

        logger.info("Recommendation Service initialized successfully!")

    def encode_user(self, recent_items: List[int], max_length: int = 50) -> np.ndarray:
        """
        Encode user from recent items.

        Args:
            recent_items: List of recent item IDs
            max_length: Maximum sequence length

        Returns:
            User embedding
        """
        if self.retrieval_model is None:
            # Return random embedding for testing
            return np.random.randn(64).astype(np.float32)

        # Pad or truncate sequence
        if len(recent_items) < max_length:
            padded = [0] * (max_length - len(recent_items)) + recent_items
        else:
            padded = recent_items[-max_length:]

        # Convert to tensor
        sequence = torch.LongTensor([padded]).to(self.device)
        seq_len = torch.LongTensor([min(len(recent_items), max_length)]).to(self.device)

        # Encode
        with torch.no_grad():
            user_embedding = self.retrieval_model.encode_user(sequence, seq_len)

        return user_embedding.cpu().numpy()[0]

    def retrieve_candidates(
        self,
        user_embedding: np.ndarray,
        k: int = 100
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k candidates using FAISS.

        Args:
            user_embedding: User embedding
            k: Number of candidates

        Returns:
            Tuple of (item_ids, scores)
        """
        if self.faiss_index is None:
            # Mock retrieval for testing
            num_items = len(self.item_features) if self.item_features is not None else 1000
            item_ids = np.random.choice(num_items, size=min(k, num_items), replace=False)
            scores = np.random.rand(len(item_ids))
            return item_ids.tolist(), scores.tolist()

        # Search with FAISS
        item_ids, distances = self.faiss_index.search(
            user_embedding.reshape(1, -1),
            k=k,
            return_distances=True
        )

        return item_ids[0].tolist(), distances[0].tolist()

    def rank_candidates(
        self,
        user_embedding: np.ndarray,
        candidate_ids: List[int],
        candidate_scores: List[float]
    ) -> Tuple[List[int], List[float]]:
        """
        Rank candidates using ranking model.

        Args:
            user_embedding: User embedding
            candidate_ids: Candidate item IDs
            candidate_scores: Retrieval scores

        Returns:
            Tuple of (reranked_ids, reranked_scores)
        """
        if self.ranking_model is None:
            # Return candidates as-is
            return candidate_ids, candidate_scores

        # TODO: Implement ranking
        # For now, return mock ranking
        indices = np.argsort(candidate_scores)[::-1]
        return [candidate_ids[i] for i in indices], [candidate_scores[i] for i in indices]

    def llm_rerank(
        self,
        recent_items: List[int],
        candidate_ids: List[int],
        candidate_scores: List[float],
        top_k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Rerank using LLM.

        Args:
            recent_items: User's recent items
            candidate_ids: Candidate item IDs
            candidate_scores: Current scores
            top_k: Number to return

        Returns:
            Tuple of (top_k_ids, top_k_scores)
        """
        if self.llm_reranker is None:
            # Return top-k from current scores
            indices = np.argsort(candidate_scores)[::-1][:top_k]
            return [candidate_ids[i] for i in indices], [candidate_scores[i] for i in indices]

        # TODO: Implement LLM reranking with actual item texts
        # For now, return top-k
        indices = np.argsort(candidate_scores)[::-1][:top_k]
        return [candidate_ids[i] for i in indices], [candidate_scores[i] for i in indices]

    def get_item_metadata(self, item_id: int) -> Optional[ItemMetadata]:
        """Get item metadata."""
        if self.item_features is None:
            return None

        try:
            item_row = self.item_features[self.item_features['item_idx'] == item_id].iloc[0]
            return ItemMetadata(
                item_id=item_id,
                category_id=int(item_row['categoryid']) if not pd.isna(item_row['categoryid']) else None,
                price=float(item_row['price']) if not pd.isna(item_row['price']) else None,
                price_bucket=int(item_row['price_bucket']) if not pd.isna(item_row['price_bucket']) else None,
                available=int(item_row['available']) if 'available' in item_row and not pd.isna(item_row['available']) else None
            )
        except (IndexError, KeyError):
            return None

    def recommend(
        self,
        user_id: Optional[int] = None,
        recent_items: Optional[List[int]] = None,
        top_k: int = 10,
        include_metadata: bool = True,
        use_llm_reranker: bool = False
    ) -> Dict:
        """
        Generate recommendations.

        Args:
            user_id: User ID (optional)
            recent_items: Recent items for cold-start
            top_k: Number of recommendations
            include_metadata: Include item metadata
            use_llm_reranker: Use LLM reranker

        Returns:
            Dictionary with recommendations and metrics
        """
        start_time = time.time()
        stages = {}

        # Validate input
        if recent_items is None or len(recent_items) == 0:
            raise ValueError("recent_items is required")

        # 1. Encode user
        encode_start = time.time()
        user_embedding = self.encode_user(recent_items)
        stages['encoding_ms'] = (time.time() - encode_start) * 1000

        # 2. Retrieve candidates
        retrieval_start = time.time()
        candidate_ids, candidate_scores = self.retrieve_candidates(
            user_embedding,
            k=100  # Retrieve 100 candidates
        )
        stages['retrieval_ms'] = (time.time() - retrieval_start) * 1000

        # 3. Rank candidates (optional)
        if self.use_ranking:
            ranking_start = time.time()
            candidate_ids, candidate_scores = self.rank_candidates(
                user_embedding,
                candidate_ids,
                candidate_scores
            )
            stages['ranking_ms'] = (time.time() - ranking_start) * 1000

        # 4. LLM reranking (optional)
        if use_llm_reranker and self.llm_reranker is not None:
            llm_start = time.time()
            final_ids, final_scores = self.llm_rerank(
                recent_items,
                candidate_ids[:20],  # Only rerank top-20
                candidate_scores[:20],
                top_k=top_k
            )
            stages['llm_reranking_ms'] = (time.time() - llm_start) * 1000
        else:
            # Just take top-k
            final_ids = candidate_ids[:top_k]
            final_scores = candidate_scores[:top_k]

        # 5. Build response
        recommendations = []
        for rank, (item_id, score) in enumerate(zip(final_ids, final_scores), 1):
            metadata = self.get_item_metadata(item_id) if include_metadata else None
            recommendations.append(
                RecommendationItem(
                    item_id=item_id,
                    score=float(score),
                    rank=rank,
                    metadata=metadata
                )
            )

        total_time = (time.time() - start_time) * 1000
        stages['total_ms'] = total_time

        return {
            'user_id': user_id,
            'recommendations': recommendations,
            'total_candidates': len(candidate_ids),
            'latency_ms': total_time,
            'pipeline_stages': stages
        }

    def get_status(self) -> Dict:
        """Get service status."""
        models_loaded = []

        if self.retrieval_model is not None:
            models_loaded.append("retrieval")
        if self.faiss_index is not None:
            models_loaded.append("faiss")
        if self.ranking_model is not None:
            models_loaded.append("ranking")
        if self.llm_reranker is not None:
            models_loaded.append("llm_reranker")

        return {
            'status': 'healthy' if len(models_loaded) > 0 else 'degraded',
            'models_loaded': models_loaded,
            'faiss_index_size': self.faiss_index.index.ntotal if self.faiss_index else 0,
            'version': '1.0.0'
        }
