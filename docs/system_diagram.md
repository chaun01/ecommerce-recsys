# System Architecture Diagram

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE                             │
│                       (Training Phase)                              │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐  │
│  │ Raw Data │───>│    Data       │───>│  Train / Val / Test Split │  │
│  │ events   │    │ Preprocessing │    │    (time-based)           │  │
│  │ items    │    │ - filtering   │    └───────────┬───────────────┘  │
│  │ category │    │ - encoding    │                │                  │
│  └──────────┘    │ - weighting   │                │                  │
│                  └──────────────┘                │                  │
│                                                  ▼                  │
│                                    ┌─────────────────────────┐      │
│                                    │   Two-Tower Model       │      │
│                                    │   Training (PyTorch)    │      │
│                                    │                         │      │
│                                    │  ┌─────────┐ ┌───────┐ │      │
│                                    │  │ User    │ │ Item  │ │      │
│                                    │  │ Tower   │ │ Tower │ │      │
│                                    │  └────┬────┘ └───┬───┘ │      │
│                                    └───────┼─────────┼──────┘      │
│                                            │         │              │
│                                            ▼         ▼              │
│                                    ┌──────────┐ ┌──────────┐       │
│                                    │ User     │ │ Item     │       │
│                                    │ Embed-   │ │ Embed-   │       │
│                                    │ dings    │ │ dings    │       │
│                                    └──────────┘ └─────┬────┘       │
│                                                       │             │
│                                                       ▼             │
│                                               ┌──────────────┐     │
│                                               │ FAISS Index  │     │
│                                               │ Building     │     │
│                                               └──────────────┘     │
│                                                       │             │
│                                                       ▼             │
│                                    ┌─────────────────────────┐      │
│                                    │   Wide & Deep Model     │      │
│                                    │   Training (Ranking)    │      │
│                                    │   - user embedding      │      │
│                                    │   - item embedding      │      │
│                                    │   - category crosses    │      │
│                                    │   - retrieval score     │      │
│                                    └─────────────────────────┘      │
│                                                                     │
│  Artifacts saved: two_tower.pt, wide_deep.pt, faiss_index.bin,     │
│                   user_embeddings.npy, item_embeddings.npy          │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                        ONLINE PIPELINE                              │
│                      (Inference Phase)                              │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐  │
│  │  Client  │───>│  FastAPI      │───>│  Step 1: RETRIEVAL        │  │
│  │  Request │    │  /recommend   │    │  - User embedding lookup  │  │
│  │          │    │               │    │  - FAISS search Top-100   │  │
│  │ visitor_ │    └──────────────┘    └───────────┬───────────────┘  │
│  │ id, top_k│                                    │                  │
│  └──────────┘                                    ▼                  │
│                                    ┌───────────────────────────┐    │
│                                    │  Step 2: RANKING           │    │
│                                    │  - Wide & Deep model       │    │
│                                    │  - Score all candidates    │    │
│                                    │  - Features: cosine sim,   │    │
│                                    │    category count, recency │    │
│                                    └───────────┬───────────────┘    │
│                                                │                    │
│                                                ▼                    │
│                                    ┌───────────────────────────┐    │
│                                    │  Step 3: LLM RERANKER     │    │
│                                    │  (Optional)               │    │
│                                    │  - User history context   │    │
│                                    │  - Item metadata          │    │
│                                    │  - Semantic reranking     │    │
│                                    └───────────┬───────────────┘    │
│                                                │                    │
│                                                ▼                    │
│                                    ┌───────────────────────────┐    │
│  ┌──────────┐                      │  Response: Top-K          │    │
│  │  Client  │<─────────────────────│  Recommendations          │    │
│  │ Response │                      │  [item_id, score, cat_id] │    │
│  └──────────┘                      └───────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
Offline:  Raw CSV ──> Preprocess ──> Train Two-Tower ──> Export Embeddings ──> Build FAISS ──> Train Ranking
Online:   User ID ──> User Embedding ──> FAISS Top-100 ──> Wide&Deep Rerank ──> (LLM Rerank) ──> Top-K Response
```

## Ablation Study Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Pipeline A:  Retrieval Only                                        │
│  User Emb ──> FAISS Top-K ──> Output                               │
│                                                                     │
│  Pipeline B:  Retrieval + Ranking                                   │
│  User Emb ──> FAISS Top-100 ──> Wide&Deep Rerank ──> Top-K Output  │
│                                                                     │
│  Pipeline C:  Retrieval + Ranking + LLM                             │
│  User Emb ──> FAISS Top-100 ──> Wide&Deep ──> LLM Rerank ──> Top-K │
│                                                                     │
│  Metrics: Recall@5, Recall@10, Recall@20, NDCG@5, NDCG@10, NDCG@20│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```
