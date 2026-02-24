# Recommendation API

FastAPI service for Next Purchase Prediction recommendation system.

## Features

- **RESTful API**: Clean HTTP endpoints
- **Async/Await**: Non-blocking I/O
- **Auto Documentation**: Swagger UI at `/docs`
- **Validation**: Pydantic models
- **Docker Ready**: Containerized deployment

## Quick Start

### 1. Run Locally

```bash
# Install FastAPI
pip install fastapi uvicorn

# Start server
python -m api.main

# Or with uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: http://localhost:8000

### 2. Run with Docker

```bash
cd docker
docker-compose up --build
```

## API Endpoints

### GET `/`
Root endpoint with API info.

```bash
curl http://localhost:8000/
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["retrieval", "faiss"],
  "faiss_index_size": 68067,
  "version": "1.0.0"
}
```

### POST `/recommend`
Get personalized recommendations.

**Request:**
```json
{
  "user_id": 12345,
  "recent_items": [101, 203, 405],
  "top_k": 10,
  "include_metadata": true,
  "use_llm_reranker": false
}
```

**Response:**
```json
{
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
    "encoding_ms": 2.3,
    "retrieval_ms": 0.8,
    "total_ms": 45.2
  }
}
```

### GET `/items/{item_id}`
Get item metadata.

```bash
curl http://localhost:8000/items/506
```

### GET `/stats`
Get system statistics.

```bash
curl http://localhost:8000/stats
```

## Usage Examples

### cURL

```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "recent_items": [101, 203, 405],
    "top_k": 10
  }'
```

### Python (requests)

```python
import requests

# Get recommendations
response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "user_id": 12345,
        "recent_items": [101, 203, 405],
        "top_k": 10,
        "include_metadata": True
    }
)

recommendations = response.json()
print(f"Got {len(recommendations['recommendations'])} recommendations")
```

### JavaScript (fetch)

```javascript
fetch('http://localhost:8000/recommend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 12345,
    recent_items: [101, 203, 405],
    top_k: 10
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

## Configuration

Update model paths in `api/main.py`:

```python
service = RecommendationService(
    retrieval_model_path="models/retrieval/checkpoints/best_model.pth",
    faiss_index_path="retrieval/indices/item_index.faiss",
    item_features_path="data/processed/item_features.parquet",
    use_ranking=False,      # Enable ranking model
    use_llm_reranker=False  # Enable LLM reranker
)
```

## API Documentation

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Request Parameters

#### RecommendationRequest
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| user_id | int | No | None | User ID |
| recent_items | List[int] | Yes | - | Recent item IDs |
| top_k | int | No | 10 | Number of recommendations |
| include_metadata | bool | No | true | Include item metadata |
| use_llm_reranker | bool | No | false | Use LLM reranker |

#### RecommendationResponse
| Field | Type | Description |
|-------|------|-------------|
| user_id | int | User ID |
| recommendations | List[Item] | Recommended items |
| total_candidates | int | Total candidates retrieved |
| latency_ms | float | Total latency |
| pipeline_stages | dict | Latency breakdown |

## Performance

### Latency Breakdown

| Stage | Latency | Description |
|-------|---------|-------------|
| Encoding | ~2ms | User embedding |
| Retrieval | <1ms | FAISS ANN search |
| Ranking | ~10ms | Wide & Deep (optional) |
| LLM Reranking | ~50ms | Transformer (optional) |
| **Total** | **~15-60ms** | End-to-end |

### Throughput

- **Without ranking**: ~200 req/s (single worker)
- **With ranking**: ~50 req/s
- **With LLM**: ~20 req/s

Scale with multiple workers:
```bash
uvicorn api.main:app --workers 4
```

## Docker Deployment

### Build Image

```bash
docker build -t recsys-api -f docker/Dockerfile .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  recsys-api
```

### Docker Compose

```bash
cd docker
docker-compose up -d
```

## Error Handling

### 400 Bad Request
- Missing required fields
- Invalid parameters

```json
{
  "error": "Validation error",
  "detail": "recent_items is required"
}
```

### 404 Not Found
- Item not found

```json
{
  "error": "Item not found",
  "detail": "Item 999999 not found"
}
```

### 500 Internal Server Error
- Model loading failed
- Inference error

```json
{
  "error": "Internal server error",
  "detail": "Failed to encode user"
}
```

### 503 Service Unavailable
- Service not initialized
- Models not loaded

```json
{
  "error": "Service not initialized",
  "detail": "Service not ready"
}
```

## Monitoring

### Logs

```bash
# View logs
tail -f logs/api.log

# With Docker
docker logs -f recsys-api
```

### Metrics

Monitor these metrics:
- Request latency (p50, p95, p99)
- Error rate
- Model inference time
- Cache hit rate

## Security

### Production Checklist

- [ ] Add API key authentication
- [ ] Rate limiting
- [ ] HTTPS/TLS
- [ ] CORS configuration
- [ ] Input sanitization
- [ ] Logging sensitive data

### Example: API Key Auth

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
```

## Troubleshooting

### Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :8000   # Windows
```

### Models not loading
- Check file paths in configuration
- Ensure models are trained and saved
- Check file permissions

### Slow response times
- Enable GPU inference
- Add caching layer (Redis)
- Use multiple workers
- Optimize batch size

## Development

### Run Tests

```bash
pytest tests/api/
```

### Hot Reload

```bash
uvicorn api.main:app --reload
```

### Debug Mode

```python
# In main.py
app = FastAPI(debug=True)
```

## Next Steps

1. Add authentication
2. Implement caching (Redis)
3. Add A/B testing
4. Monitor with Prometheus
5. Load testing (Locust)

## License

MIT License
