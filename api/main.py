"""
FastAPI application for recommendation system.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    RecommendationRequest,
    RecommendationResponse,
    HealthResponse,
    ErrorResponse
)
from api.service import RecommendationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instance
service: RecommendationService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the application."""
    global service

    # Startup
    logger.info("Starting Recommendation API...")

    # Initialize service
    # Note: Update these paths based on your trained models
    service = RecommendationService(
        retrieval_model_path=None,  # "models/retrieval/checkpoints/best_model.pth"
        faiss_index_path=None,      # "retrieval/indices/item_index.faiss"
        item_features_path="data/processed/item_features.parquet",
        use_ranking=False,
        use_llm_reranker=False,
        device="cpu"
    )

    logger.info("Recommendation API started successfully!")

    yield

    # Shutdown
    logger.info("Shutting down Recommendation API...")


# Create FastAPI app
app = FastAPI(
    title="Next Purchase Prediction API",
    description="E-commerce recommendation system API with retrieval, ranking, and optional LLM reranking",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Error processing request: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Next Purchase Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and loaded models.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    status = service.get_status()
    return HealthResponse(**status)


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Get personalized recommendations.

    Args:
        request: Recommendation request with user info

    Returns:
        List of recommended items with scores

    Example:
        ```
        POST /recommend
        {
            "user_id": 12345,
            "recent_items": [101, 203, 405],
            "top_k": 10,
            "include_metadata": true,
            "use_llm_reranker": false
        }
        ```
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate request
    if not request.recent_items or len(request.recent_items) == 0:
        raise HTTPException(
            status_code=400,
            detail="recent_items is required and must not be empty"
        )

    try:
        # Get recommendations
        result = service.recommend(
            user_id=request.user_id,
            recent_items=request.recent_items,
            top_k=request.top_k,
            include_metadata=request.include_metadata,
            use_llm_reranker=request.use_llm_reranker
        )

        return RecommendationResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/items/{item_id}")
async def get_item(item_id: int):
    """
    Get item metadata.

    Args:
        item_id: Item ID

    Returns:
        Item metadata
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    metadata = service.get_item_metadata(item_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    return metadata


@app.get("/stats")
async def get_stats():
    """
    Get system statistics.

    Returns:
        Statistics about the recommendation system
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = {
        "total_items": len(service.item_features) if service.item_features is not None else 0,
        "faiss_index_size": service.faiss_index.index.ntotal if service.faiss_index else 0,
        "models": service.get_status()['models_loaded']
    }

    return stats


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
