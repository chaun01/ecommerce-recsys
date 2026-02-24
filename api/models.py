"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""

    user_id: Optional[int] = Field(None, description="User ID (if known)")
    recent_items: Optional[List[int]] = Field(
        None,
        description="Recent item IDs user interacted with (for cold-start)"
    )
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    include_metadata: bool = Field(True, description="Include item metadata in response")
    use_llm_reranker: bool = Field(False, description="Use LLM reranker (slower but better)")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "recent_items": [101, 203, 405],
                "top_k": 10,
                "include_metadata": True,
                "use_llm_reranker": False
            }
        }


class ItemMetadata(BaseModel):
    """Item metadata."""

    item_id: int
    category_id: Optional[int] = None
    price: Optional[float] = None
    price_bucket: Optional[int] = None
    available: Optional[int] = None


class RecommendationItem(BaseModel):
    """Single recommendation item."""

    item_id: int = Field(..., description="Item ID")
    score: float = Field(..., description="Recommendation score (0-1)")
    rank: int = Field(..., description="Rank in recommendation list (1-based)")
    metadata: Optional[ItemMetadata] = Field(None, description="Item metadata")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""

    user_id: Optional[int] = Field(None, description="User ID")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommended items")
    total_candidates: int = Field(..., description="Total candidates retrieved")
    latency_ms: float = Field(..., description="Total latency in milliseconds")
    pipeline_stages: dict = Field(..., description="Latency breakdown by stage")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "recommendations": [
                    {
                        "item_id": 506,
                        "score": 0.94,
                        "rank": 1,
                        "metadata": {
                            "item_id": 506,
                            "category_id": 1338,
                            "price": 15000.0,
                            "price_bucket": 2
                        }
                    }
                ],
                "total_candidates": 100,
                "latency_ms": 45.2,
                "pipeline_stages": {
                    "retrieval_ms": 0.8,
                    "ranking_ms": 12.4,
                    "total_ms": 45.2
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    faiss_index_size: int = Field(..., description="Number of items in FAISS index")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
