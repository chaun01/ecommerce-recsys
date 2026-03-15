# E-commerce Recommendation System — Next Purchase Prediction

An end-to-end recommendation system for predicting the next product a user is most likely to purchase, built on the RetailRocket e-commerce dataset.

## System Architecture

```
Offline Pipeline (Training Phase)
──────────────────────────────────
Raw Data → Preprocessing → Two-Tower Training → Export Embeddings → Build FAISS Index → Train Ranking Model

Online Pipeline (Inference Phase)
─────────────────────────────────
User ID → User Embedding → FAISS Top-100 → Wide&Deep Rerank → (LLM Rerank) → Top-K Response
```

### Detailed Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE                          │
│                                                             │
│  Raw CSV ──▶ Preprocess ──▶ Train Two-Tower ──▶ Embeddings  │
│                                    │                        │
│                                    ▼                        │
│                             FAISS Index Build               │
│                                    │                        │
│                                    ▼                        │
│                          Train Wide & Deep                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    ONLINE PIPELINE                           │
│                                                             │
│  POST /recommend                                            │
│       │                                                     │
│       ▼                                                     │
│  User Embedding Lookup                                      │
│       │                                                     │
│       ▼                                                     │
│  FAISS Retrieval (Top-100 candidates)                       │
│       │                                                     │
│       ▼                                                     │
│  Wide & Deep Ranking (score + rerank)                       │
│       │                                                     │
│       ▼                                                     │
│  (Optional) LLM Reranker                                    │
│       │                                                     │
│       ▼                                                     │
│  Top-K Recommendations Response                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
recsys/
├── data/
│   ├── events.csv                 # User interactions (view/addtocart/transaction)
│   ├── item_properties_part1.csv  # Item metadata
│   ├── item_properties_part2.csv
│   ├── category_tree.csv          # Category hierarchy
│   └── preprocess.py              # Data preprocessing + time-based split
├── retrieval/
│   ├── two_tower.py               # Two-Tower neural retrieval model
│   ├── dataset.py                 # Training dataset + collate function
│   ├── train.py                   # Retrieval training + embedding export
│   └── faiss_index.py             # FAISS index build/search
├── ranking/
│   ├── wide_deep.py               # Wide & Deep ranking model
│   ├── dataset.py                 # Ranking dataset with features
│   ├── train.py                   # Ranking model training
│   └── llm_reranker.py            # LLM-based reranker (OpenAI)
├── api/
│   └── main.py                    # FastAPI REST API
├── evaluation/
│   ├── metrics.py                 # Recall@K, NDCG@K
│   ├── evaluate.py                # Ablation study
│   └── results.txt                # Evaluation results
├── models/                        # Saved model artifacts (generated)
├── docs/
│   └── system_diagram.md          # System architecture diagram
├── run_pipeline.py                # Full pipeline runner
├── requirements.txt
└── Dockerfile
```

## Dataset

**RetailRocket E-commerce Dataset**

| File | Records | Description |
|------|---------|-------------|
| `events.csv` | ~2.76M | User events: view (2.66M), addtocart (69K), transaction (22K) |
| `item_properties_part1.csv` | ~11M | Item properties (category, attributes) |
| `item_properties_part2.csv` | ~9.3M | Item properties continued |
| `category_tree.csv` | ~1.7K | Category hierarchy |

After preprocessing (filtering cold-start users/items): **66,146 users**, **36,184 items**, **782,330 interactions**.

## Pipeline Details

### 1. Data Preprocessing (`data/preprocess.py`)
- Implicit feedback weighting: view=1, addtocart=3, transaction=5
- Iterative cold-start filtering (min 5 interactions per user/item)
- Time-based train/val/test split (80/10/10)
- ID encoding and category extraction

### 2. Retrieval — Two-Tower Model (`retrieval/two_tower.py`)
- **User Tower**: MLP on weighted average of historical item embeddings
- **Item Tower**: Item embedding + category embedding → MLP
- Contrastive learning with in-batch + random negatives
- Exports normalized user/item embeddings (dim=64)

### 3. FAISS ANN Search (`retrieval/faiss_index.py`)
- Inner product index (cosine similarity on normalized embeddings)
- Retrieves Top-100 candidate items per user query

### 4. Ranking — Wide & Deep (`ranking/wide_deep.py`)
- **Wide part**: Category feature crosses (linear)
- **Deep part**: User embedding + item embedding + category embedding + features (retrieval score, category interaction count, recency) → MLP
- Binary cross-entropy loss on positive/negative candidates

### 5. LLM Reranker (`ranking/llm_reranker.py`)
- Uses OpenAI API to rescore candidates
- Input: user history + candidate item metadata
- Output: semantically reranked list

### 6. API Serving (`api/main.py`)
- **Framework**: FastAPI
- **Endpoint**: `POST /recommend` — full pipeline inference
- **Health check**: `GET /health`
- **Containerized** with Docker

## Evaluation Results

### Ablation Study

| Metric | Retrieval Only | Retrieval + Ranking |
|--------|---------------|-------------------|
| Recall@5 | 0.0360 | **0.0501** |
| NDCG@5 | 0.0312 | **0.0503** |
| Recall@10 | 0.0399 | **0.0510** |
| NDCG@10 | 0.0325 | **0.0504** |
| Recall@20 | 0.0444 | **0.0525** |
| NDCG@20 | 0.0338 | **0.0508** |

The ranking model improves Recall@5 by ~39% and NDCG@5 by ~61% over retrieval-only.

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python run_pipeline.py
```

### Run Individual Stages
```bash
python run_pipeline.py preprocess   # Data preprocessing
python run_pipeline.py retrieval    # Train Two-Tower model
python run_pipeline.py faiss        # Build FAISS index
python run_pipeline.py ranking      # Train Wide & Deep model
python run_pipeline.py evaluate     # Run evaluation
```

### Start API Server
```bash
python run_pipeline.py serve
# or
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t recsys .
docker run -p 8000:8000 recsys
```

### API Usage Example
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"visitor_id": 257597, "top_k": 10, "use_ranking": true, "use_llm": false}'
```

## Tech Stack

- **Python 3.12**
- **PyTorch** — model training (Two-Tower, Wide & Deep)
- **FAISS** — approximate nearest neighbor search
- **FastAPI** — REST API serving
- **Docker** — containerization
- **scikit-learn** — preprocessing utilities
