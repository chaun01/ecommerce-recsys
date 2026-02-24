# System Architecture - Next Purchase Prediction RecSys

## Overview
End-to-end recommendation system for predicting next purchase in e-commerce using RetailRocket dataset with implicit feedback (view, addtocart, transaction events).

---

## Data Understanding

### Dataset Files
1. **events.csv** - User behavior data
   - `timestamp`: Event time
   - `visitorid`: User identifier
   - `event`: Action type (view, addtocart, transaction)
   - `itemid`: Product identifier
   - `transactionid`: Transaction ID (for purchases)

2. **item_properties.csv** - Product metadata
   - `timestamp`: Property update time
   - `itemid`: Product identifier
   - `property`: Property type (categoryid, available, price codes, etc.)
   - `value`: Property value

3. **category_tree.csv** - Category hierarchy
   - `categoryid`: Category identifier
   - `parentid`: Parent category

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OFFLINE PIPELINE                             │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   Raw Data       │
│ - events.csv     │
│ - items.csv      │
│ - categories.csv │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│              DATA PREPROCESSING                                   │
│ - Parse timestamps, filter invalid events                        │
│ - Build user-item interaction matrix                             │
│ - Extract item features (category, price, availability)          │
│ - Create implicit feedback signals (view=1, addtocart=2, buy=3)  │
│ - Time-based train/val/test split                                │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────┬──────────────────────┐
         ▼                                 ▼                      ▼
┌────────────────────┐          ┌──────────────────┐    ┌─────────────┐
│ RETRIEVAL TRAINING │          │ RANKING TRAINING │    │ FAISS INDEX │
│                    │          │                  │    │   BUILDER   │
│ Two-Tower Model:   │          │ Wide & Deep:     │    │             │
│ - User Tower       │          │ - Wide: cross    │    │ - Item      │
│   (session clicks) │          │   features       │    │   embeddings│
│ - Item Tower       │          │ - Deep: user/    │    │ - ANN index │
│   (product emb)    │          │   item embeddings│    │             │
│                    │          │                  │    │             │
│ Loss: Contrastive  │          │ Loss: BPR/BCE    │    │             │
└────────┬───────────┘          └─────────┬────────┘    └──────┬──────┘
         │                                │                     │
         ▼                                ▼                     │
┌─────────────────┐           ┌──────────────────┐             │
│ User Embeddings │           │ Ranking Model    │             │
│ Item Embeddings │           │ Weights (.pth)   │             │
│   (.npy)        │           └──────────────────┘             │
└─────────┬───────┘                                             │
          │                                                     │
          └─────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │  EVALUATION PIPELINE   │
                        │  - Recall@K            │
                        │  - NDCG@K              │
                        │  - Ablation study      │
                        └────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          ONLINE PIPELINE                             │
└─────────────────────────────────────────────────────────────────────┘

         ┌──────────────────┐
         │  API REQUEST     │
         │  POST /recommend │
         │  {user_id,       │
         │   recent_items}  │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────────────┐
         │   USER EMBEDDING         │
         │   - Lookup from cache or │
         │   - Encode recent session│
         └────────┬─────────────────┘
                  │
                  ▼
         ┌──────────────────────────┐
         │  RETRIEVAL STAGE         │
         │  - FAISS ANN search      │
         │  - Top-N candidates      │
         │    (e.g., N=100)         │
         └────────┬─────────────────┘
                  │
                  ▼
         ┌──────────────────────────┐
         │   RANKING STAGE          │
         │   - Wide & Deep model    │
         │   - Score each candidate │
         │   - Sort by score        │
         └────────┬─────────────────┘
                  │
                  ▼
         ┌──────────────────────────┐
         │  OPTIONAL: LLM RERANKER  │
         │  - Product titles + hist │
         │  - Semantic relevance    │
         └────────┬─────────────────┘
                  │
                  ▼
         ┌──────────────────────────┐
         │   RESPONSE               │
         │   {recommendations: [    │
         │     {item_id, score}]}   │
         └──────────────────────────┘
```

---

## Component Details

### 1. Data Preprocessing (`data/`)

**Input:**
- Raw CSV files from RetailRocket dataset

**Processing Steps:**
1. **Event Processing**
   - Parse timestamps to datetime
   - Filter out invalid events
   - Weight events: view=1, addtocart=2, transaction=3
   - Create user-item interaction sequences

2. **Item Feature Engineering**
   - Extract categoryid from item_properties
   - Parse price information (n*.000 format)
   - Get availability status
   - Build category hierarchy from category_tree

3. **Data Splitting**
   - Time-based split (e.g., 70% train, 15% val, 15% test)
   - Ensure no future data leaks
   - Keep temporal order for sequential models

**Output:**
- `train_interactions.parquet`
- `val_interactions.parquet`
- `test_interactions.parquet`
- `item_features.parquet`
- `user_sequences.parquet`

---

### 2. Retrieval Model (`retrieval/`)

**Model Architecture: Two-Tower Neural Network**

```python
User Tower:
  Input: recent_item_ids [sequence of last N items]
  ↓
  Embedding Layer (item_id → dense vector)
  ↓
  Aggregation (mean/attention pooling)
  ↓
  MLP [256, 128, 64]
  ↓
  User Embedding (64-dim)

Item Tower:
  Input: item_id + category + features
  ↓
  Embedding Layer
  ↓
  MLP [256, 128, 64]
  ↓
  Item Embedding (64-dim)

Loss: Contrastive Learning (InfoNCE)
  - Positive: items user interacted with
  - Negative: random sampling + hard negatives
```

**Training:**
- Framework: PyTorch
- Batch size: 512-1024
- Optimizer: AdamW
- Learning rate: 1e-3 with scheduler
- Epochs: 10-20

**Output:**
- User embeddings: `user_embeddings.npy` (shape: [num_users, 64])
- Item embeddings: `item_embeddings.npy` (shape: [num_items, 64])
- Model checkpoint: `two_tower_model.pth`

---

### 3. FAISS Index (`retrieval/`)

**Index Type:** IndexIVFFlat or IndexHNSWFlat
- For datasets <1M items: HNSW (better accuracy)
- For larger datasets: IVF (faster search)

**Building:**
```python
import faiss
index = faiss.IndexHNSWFlat(embedding_dim, M=32)
index.add(item_embeddings)
faiss.write_index(index, "item_index.faiss")
```

**Search:**
- Top-K retrieval: K=100 candidates per user
- Latency target: <10ms per query

---

### 4. Ranking Model (`ranking/`)

**Model Architecture: Wide & Deep**

```python
Wide Component:
  Input: Cross features
    - user_category_hist × item_category
    - price_bucket × user_price_preference
    - recency_bucket × item_popularity
  ↓
  Linear Layer
  ↓
  Wide Output

Deep Component:
  Input: Dense features
    - user_embedding (64-dim)
    - item_embedding (64-dim)
    - interaction features (time, frequency)
  ↓
  MLP [128, 64, 32]
  ↓
  Deep Output

Combined:
  sigmoid(Wide Output + Deep Output)
```

**Training:**
- Positive samples: actual interactions
- Negative samples: retrieved but not interacted
- Loss: Binary Cross-Entropy or BPR
- Metrics: AUC, Logloss

**Alternative: DeepFM**
- Replaces Wide component with FM layer
- Better feature interaction modeling

**Output:**
- Trained model: `ranking_model.pth`

---

### 5. Optional: LLM Reranker (`reranking/`)

**Model:** Small LLM (e.g., BERT, T5, or Llama-3B)

**Input Format:**
```
User History: [item_1_title, item_2_title, ...]
Candidates: [candidate_1_title, candidate_2_title, ...]
Task: Rank candidates by purchase likelihood
```

**Approach:**
- Pointwise: Score each candidate independently
- Pairwise: Compare candidate pairs
- Listwise: Rerank entire list

**Integration:**
- Rerank top-20 from ranking model
- Final output: top-10 recommendations

---

### 6. API Service (`api/`)

**Framework:** FastAPI + Uvicorn

**Endpoints:**

#### POST /recommend
```json
Request:
{
  "user_id": "12345",
  "recent_items": [101, 203, 405],  // optional
  "top_k": 10,
  "include_metadata": true
}

Response:
{
  "user_id": "12345",
  "recommendations": [
    {
      "item_id": 506,
      "score": 0.94,
      "category": 1338,
      "metadata": {...}
    },
    ...
  ],
  "latency_ms": 45
}
```

#### GET /health
```json
{
  "status": "healthy",
  "models_loaded": ["retrieval", "ranking"],
  "faiss_index_size": 500000
}
```

**Service Architecture:**
```python
class RecommendationService:
    def __init__(self):
        self.retrieval_model = load_two_tower()
        self.ranking_model = load_wide_deep()
        self.faiss_index = faiss.read_index()
        self.item_features = load_item_features()

    def recommend(self, user_id, recent_items, top_k):
        # 1. Get user embedding
        user_emb = self.encode_user(recent_items)

        # 2. Retrieve candidates
        candidates = self.faiss_index.search(user_emb, k=100)

        # 3. Rank candidates
        scores = self.ranking_model(user_emb, candidates)

        # 4. Return top-k
        return sorted(zip(candidates, scores))[:top_k]
```

**Dockerization:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 7. Evaluation (`evaluation/`)

**Offline Metrics:**

1. **Retrieval Metrics**
   - Recall@K (K=10, 20, 50, 100)
   - MRR (Mean Reciprocal Rank)
   - Coverage (% of items recommended)

2. **Ranking Metrics**
   - NDCG@K (K=5, 10, 20)
   - Precision@K
   - MAP (Mean Average Precision)

**Ablation Study:**
```
Configuration 1: Retrieval only (Two-Tower + FAISS)
Configuration 2: Retrieval + Ranking (+ Wide & Deep)
Configuration 3: Retrieval + Ranking + LLM Reranker

Compare:
- Recall@10, NDCG@10
- Latency (p50, p95, p99)
- Model size and complexity
```

**Evaluation Script:**
```python
# evaluation/evaluate.py
def evaluate_retrieval(model, test_data):
    recalls = []
    for user, ground_truth in test_data:
        predictions = model.recommend(user, k=100)
        recalls.append(recall_at_k(predictions, ground_truth, k=10))
    return np.mean(recalls)
```

---

## Project Structure

```
recsys/
├── data/
│   ├── raw/
│   │   ├── events.csv
│   │   ├── item_properties_part1.csv
│   │   ├── item_properties_part2.csv
│   │   └── category_tree.csv
│   ├── processed/
│   │   ├── train_interactions.parquet
│   │   ├── val_interactions.parquet
│   │   ├── test_interactions.parquet
│   │   ├── item_features.parquet
│   │   └── user_sequences.parquet
│   └── preprocessing.py
│
├── models/
│   ├── retrieval/
│   │   ├── two_tower.py
│   │   ├── train_retrieval.py
│   │   └── checkpoints/
│   │       ├── two_tower_model.pth
│   │       ├── user_embeddings.npy
│   │       └── item_embeddings.npy
│   │
│   ├── ranking/
│   │   ├── wide_deep.py
│   │   ├── train_ranking.py
│   │   └── checkpoints/
│   │       └── ranking_model.pth
│   │
│   └── reranking/  # Optional
│       ├── llm_reranker.py
│       └── train_reranker.py
│
├── retrieval/
│   ├── faiss_index.py
│   ├── build_index.py
│   └── indices/
│       └── item_index.faiss
│
├── api/
│   ├── main.py
│   ├── service.py
│   ├── models.py (Pydantic schemas)
│   └── utils.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   └── ablation_study.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_training.ipynb
│   └── 03_ranking_training.ipynb
│
├── configs/
│   ├── retrieval_config.yaml
│   └── ranking_config.yaml
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt
├── README.md
└── ARCHITECTURE.md (this file)
```

---

## Technology Stack

### Core ML/DL
- **PyTorch** 2.0+: Model training
- **FAISS** (GPU version if available): ANN search
- **NumPy** / **Pandas**: Data processing
- **Scikit-learn**: Evaluation metrics

### API & Deployment
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Request/response validation
- **Docker**: Containerization

### Optional
- **Hugging Face Transformers**: LLM reranker
- **MLflow** / **Weights & Biases**: Experiment tracking
- **Ray** / **Joblib**: Parallel processing

---

## Training Pipeline Workflow

### Phase 1: Data Preparation
```bash
python data/preprocessing.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --train_ratio 0.7 \
  --val_ratio 0.15
```

### Phase 2: Retrieval Training
```bash
python models/retrieval/train_retrieval.py \
  --config configs/retrieval_config.yaml \
  --data_dir data/processed \
  --output_dir models/retrieval/checkpoints
```

### Phase 3: Build FAISS Index
```bash
python retrieval/build_index.py \
  --embeddings models/retrieval/checkpoints/item_embeddings.npy \
  --index_type HNSW \
  --output retrieval/indices/item_index.faiss
```

### Phase 4: Ranking Training
```bash
python models/ranking/train_ranking.py \
  --config configs/ranking_config.yaml \
  --retrieval_ckpt models/retrieval/checkpoints \
  --output_dir models/ranking/checkpoints
```

### Phase 5: Evaluation
```bash
python evaluation/evaluate.py \
  --test_data data/processed/test_interactions.parquet \
  --retrieval_model models/retrieval/checkpoints \
  --ranking_model models/ranking/checkpoints \
  --output_dir evaluation/results
```

---

## Inference Pipeline

### Startup (Service Initialization)
1. Load Two-Tower model
2. Load user/item embeddings
3. Load FAISS index
4. Load Ranking model
5. Load item metadata
6. Warmup (dummy requests)

### Request Flow
```
User Request
    ↓
[1] Parse user_id + recent_items (5ms)
    ↓
[2] Encode user → user_embedding (10ms)
    ↓
[3] FAISS search → 100 candidates (8ms)
    ↓
[4] Ranking model → scores (15ms)
    ↓
[5] Sort + filter → top-K (2ms)
    ↓
[6] Attach metadata (5ms)
    ↓
Response (Total: ~45ms)
```

---

## Key Design Decisions

### 1. Implicit Feedback Weighting
- View = 1.0
- Add-to-cart = 2.0
- Transaction = 3.0

**Rationale:** Stronger signal for purchase intent

### 2. Two-Tower for Retrieval
**Pros:**
- Fast inference (precompute item embeddings)
- Scalable to millions of items
- Works well with FAISS

**Cons:**
- Cannot model complex user-item interactions

### 3. Wide & Deep for Ranking
**Pros:**
- Combines memorization (wide) + generalization (deep)
- Proven in production (Google Play, YouTube)

**Alternatives:**
- DeepFM: Better feature interactions
- xDeepFM: More expressive

### 4. Time-based Split
**Rationale:**
- Simulates real production scenario
- Prevents data leakage
- Tests model's ability to predict future

---

## Success Metrics

### Offline
- **Recall@10 > 0.15** (retrieval stage)
- **NDCG@10 > 0.20** (after ranking)
- **Latency p95 < 100ms**

### Ablation Targets
- Ranking should improve NDCG by 20-30% over retrieval-only
- LLM reranker should improve NDCG by 5-10% (if implemented)

---

## Next Steps

1. Implement data preprocessing pipeline
2. Train Two-Tower retrieval model
3. Build FAISS index
4. Train Wide & Deep ranking model
5. Develop FastAPI service
6. Run evaluation & ablation study
7. Dockerize application
8. Document results and insights

---

## References
- RetailRocket RecSys Dataset: Kaggle
- Two-Tower Models: YouTube DNN (Covington et al., 2016)
- Wide & Deep: Google (Cheng et al., 2016)
- FAISS: Facebook AI Research
