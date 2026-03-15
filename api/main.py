"""
FastAPI REST API for the recommendation system.
Flow: input user → retrieval (FAISS) → ranking (Wide & Deep) → optional LLM rerank → Top-K
"""

import os
import sys
import pickle
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.two_tower import TwoTowerModel
from retrieval.faiss_index import load_index, retrieve_top_k
from retrieval.train import compute_user_hist_emb
from ranking.wide_deep import WideAndDeep
from ranking.llm_reranker import rerank_with_llm

app = FastAPI(title="E-commerce Recommendation System", version="1.0.0")

# Global model state
state = {}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
def load_models():
    """Load all models and data on startup."""
    print("Loading models...")

    # Load encoders
    with open(os.path.join(DATA_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    state["user_enc"] = encoders["user_enc"]
    state["item_enc"] = encoders["item_enc"]

    # Load item metadata
    with open(os.path.join(DATA_DIR, "item_meta.pkl"), "rb") as f:
        state["item_meta"] = pickle.load(f)

    # Load user histories
    with open(os.path.join(DATA_DIR, "user_histories.pkl"), "rb") as f:
        state["user_histories"] = pickle.load(f)

    # Load Two-Tower model
    with open(os.path.join(MODEL_DIR, "two_tower_config.pkl"), "rb") as f:
        tt_config = pickle.load(f)
    two_tower = TwoTowerModel(**tt_config).to(DEVICE)
    two_tower.load_state_dict(torch.load(os.path.join(MODEL_DIR, "two_tower.pt"),
                                          map_location=DEVICE, weights_only=True))
    two_tower.eval()
    state["two_tower"] = two_tower

    # Load FAISS index
    state["faiss_index"] = load_index()

    # Load item embeddings
    state["item_embeddings"] = np.load(os.path.join(MODEL_DIR, "item_embeddings.npy"))

    # Load user embeddings
    state["user_embeddings"] = np.load(os.path.join(MODEL_DIR, "user_embeddings.npy"))

    # Load Wide & Deep model
    with open(os.path.join(MODEL_DIR, "wide_deep_config.pkl"), "rb") as f:
        wd_config = pickle.load(f)
    wide_deep = WideAndDeep(**wd_config).to(DEVICE)
    wide_deep.load_state_dict(torch.load(os.path.join(MODEL_DIR, "wide_deep.pt"),
                                          map_location=DEVICE, weights_only=True))
    wide_deep.eval()
    state["wide_deep"] = wide_deep

    print("All models loaded!")


# ---- Request / Response schemas ----

class RecommendRequest(BaseModel):
    visitor_id: int = Field(..., description="Original visitor ID from the dataset")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    use_ranking: bool = Field(default=True, description="Apply ranking model")
    use_llm: bool = Field(default=False, description="Apply LLM reranking")
    retrieval_candidates: int = Field(default=100, ge=10, le=500,
                                       description="Number of retrieval candidates")


class RecommendItem(BaseModel):
    item_id: int
    score: float
    category_id: int


class RecommendResponse(BaseModel):
    visitor_id: int
    recommendations: list[RecommendItem]
    pipeline: str


# ---- Endpoints ----

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(state)}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Full recommendation pipeline:
    1. Map visitor_id to internal user_idx
    2. Retrieve Top-N candidates via FAISS
    3. Rank candidates with Wide & Deep
    4. Optionally rerank with LLM
    """
    # Map visitor ID
    try:
        user_idx = int(np.where(state["user_enc"].classes_ == req.visitor_id)[0][0])
    except (IndexError, ValueError):
        raise HTTPException(status_code=404, detail=f"Visitor {req.visitor_id} not found")

    # Get user embedding
    user_emb = state["user_embeddings"][user_idx].reshape(1, -1).astype(np.float32)

    # Step 1: Retrieval via FAISS
    ret_scores, ret_indices = retrieve_top_k(
        state["faiss_index"], user_emb, top_k=req.retrieval_candidates
    )
    ret_scores = ret_scores[0]
    ret_indices = ret_indices[0]

    # Filter invalid indices
    valid_mask = ret_indices >= 0
    ret_scores = ret_scores[valid_mask]
    ret_indices = ret_indices[valid_mask]

    pipeline = "retrieval"

    if req.use_ranking and len(ret_indices) > 0:
        # Step 2: Ranking with Wide & Deep
        pipeline = "retrieval+ranking"
        item_meta = state["item_meta"]
        cat_map = dict(zip(item_meta["item_idx"], item_meta["cat_idx"]))

        with torch.no_grad():
            user_emb_t = torch.tensor(user_emb, dtype=torch.float32).expand(len(ret_indices), -1).to(DEVICE)
            item_ids_t = torch.tensor(ret_indices, dtype=torch.long).to(DEVICE)
            cat_ids_t = torch.tensor([cat_map.get(int(i), 0) for i in ret_indices], dtype=torch.long).to(DEVICE)
            ret_scores_t = torch.tensor(ret_scores, dtype=torch.float32).to(DEVICE)

            # Compute extra features
            user_hist = state["user_histories"].get(user_idx, [])
            user_cats = [cat_map.get(h[0], 0) for h in user_hist]
            cat_counts = {}
            for c in user_cats:
                cat_counts[c] = cat_counts.get(c, 0) + 1

            evt_counts = torch.tensor(
                [min(cat_counts.get(cat_map.get(int(i), 0), 0), 50) / 50.0 for i in ret_indices],
                dtype=torch.float32
            ).to(DEVICE)
            recency = torch.ones(len(ret_indices), dtype=torch.float32).to(DEVICE)

            rank_scores = state["wide_deep"](
                user_emb_t, item_ids_t, cat_ids_t, ret_scores_t, evt_counts, recency
            )
            rank_scores = rank_scores.cpu().numpy()

        # Sort by ranking score
        sorted_idx = np.argsort(-rank_scores)
        ret_indices = ret_indices[sorted_idx]
        ret_scores = rank_scores[sorted_idx]

    # Take top_k
    top_indices = ret_indices[:req.top_k]
    top_scores = ret_scores[:req.top_k]

    if req.use_llm and len(top_indices) > 0:
        pipeline += "+llm"
        # Build metadata for LLM
        item_meta = state["item_meta"]
        cat_map = dict(zip(item_meta["item_idx"], item_meta["categoryid"]))

        user_hist = state["user_histories"].get(user_idx, [])
        hist_items = [{"title": f"Item-{h[0]}", "category": str(cat_map.get(h[0], "Unknown"))}
                      for h in user_hist[-10:]]

        candidates = [
            {"id": int(idx), "title": f"Item-{idx}",
             "category": str(cat_map.get(int(idx), "Unknown")),
             "score": float(score)}
            for idx, score in zip(top_indices, top_scores)
        ]

        reranked_ids = rerank_with_llm(hist_items, candidates, top_k=req.top_k)
        # Rebuild ordered results
        id_to_score = {int(idx): float(s) for idx, s in zip(top_indices, top_scores)}
        top_indices = np.array(reranked_ids)
        top_scores = np.array([id_to_score.get(rid, 0.0) for rid in reranked_ids])

    # Build response
    item_meta = state["item_meta"]
    cat_map = dict(zip(item_meta["item_idx"], item_meta["categoryid"]))
    item_enc = state["item_enc"]

    recommendations = []
    for idx, score in zip(top_indices, top_scores):
        idx = int(idx)
        original_item_id = int(item_enc.classes_[idx]) if idx < len(item_enc.classes_) else idx
        recommendations.append(RecommendItem(
            item_id=original_item_id,
            score=round(float(score), 4),
            category_id=int(cat_map.get(idx, -1)),
        ))

    return RecommendResponse(
        visitor_id=req.visitor_id,
        recommendations=recommendations,
        pipeline=pipeline,
    )
