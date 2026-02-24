# Next Purchase Prediction - E-commerce RecSys

End-to-end recommendation system for predicting next purchase in e-commerce using RetailRocket dataset.

## Project Overview

This project implements a production-style recommendation pipeline with:
- **Retrieval Model**: Two-Tower neural network for candidate generation
- **Ranking Model**: Wide & Deep for precise scoring
- **FAISS**: Approximate nearest neighbor search for fast retrieval
- **REST API**: FastAPI service for online inference
- **Evaluation**: Comprehensive offline metrics and ablation studies

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design and component specifications.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing

```bash
python data/preprocessing.py \
  --input_dir data \
  --output_dir data/processed \
  --train_ratio 0.7 \
  --val_ratio 0.15
```

### 3. Train Retrieval Model

```bash
python models/retrieval/train_retrieval.py \
  --config configs/retrieval_config.yaml
```

### 4. Build FAISS Index

```bash
python retrieval/build_index.py \
  --embeddings models/retrieval/checkpoints/item_embeddings.npy \
  --output retrieval/indices/item_index.faiss
```

### 5. Train Ranking Model

```bash
python models/ranking/train_ranking.py \
  --config configs/ranking_config.yaml
```

### 6. Run API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:

```bash
cd docker
docker-compose up --build
```

### 7. Test API

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "12345",
    "recent_items": [101, 203, 405],
    "top_k": 10
  }'
```

## Project Structure

```
recsys/
├── data/                      # Data files and preprocessing
├── models/                    # Model implementations
│   ├── retrieval/            # Two-Tower model
│   └── ranking/              # Wide & Deep model
├── retrieval/                # FAISS index management
├── api/                      # FastAPI service
├── evaluation/               # Metrics and evaluation
├── configs/                  # Configuration files
├── docker/                   # Docker files
└── notebooks/                # Jupyter notebooks
```

## Dataset

**RetailRocket RecSys Dataset**
- `events.csv`: User behavior (view, addtocart, transaction)
- `item_properties.csv`: Product metadata
- `category_tree.csv`: Category hierarchy

## Performance Targets

### Offline Metrics
- **Recall@10**: > 0.15 (retrieval)
- **NDCG@10**: > 0.20 (ranking)
- **Latency p95**: < 100ms

### Ablation Study
Compare performance of:
1. Retrieval only
2. Retrieval + Ranking
3. Retrieval + Ranking + LLM (optional)

## API Endpoints

### POST /recommend
Get personalized recommendations for a user.

**Request:**
```json
{
  "user_id": "12345",
  "recent_items": [101, 203, 405],
  "top_k": 10,
  "include_metadata": true
}
```

**Response:**
```json
{
  "user_id": "12345",
  "recommendations": [
    {
      "item_id": 506,
      "score": 0.94,
      "category": 1338
    }
  ],
  "latency_ms": 45
}
```

### GET /health
Check service health status.

## Development

### Run Tests
```bash
pytest tests/
```

### Evaluation
```bash
python evaluation/evaluate.py \
  --test_data data/processed/test_interactions.parquet \
  --retrieval_model models/retrieval/checkpoints \
  --ranking_model models/ranking/checkpoints
```

### Ablation Study
```bash
python evaluation/ablation_study.py \
  --config configs/evaluation_config.yaml
```

## Tech Stack

- **PyTorch**: Model training
- **FAISS**: ANN search
- **FastAPI**: API framework
- **Docker**: Containerization
- **NumPy/Pandas**: Data processing

## References

- Two-Tower Models: [YouTube DNN (Covington et al., 2016)](https://research.google/pubs/pub45530/)
- Wide & Deep: [Google (Cheng et al., 2016)](https://arxiv.org/abs/1606.07792)
- FAISS: [Facebook AI Research](https://github.com/facebookresearch/faiss)

## License

MIT License

## Author

Built as part of Recommendation Systems course project.
