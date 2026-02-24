# LLM-based Reranker

Optional component that uses pre-trained language models to rerank candidates based on semantic relevance.

## Overview

The LLM reranker improves ranking by understanding semantic similarity between user history and candidate items using natural language descriptions.

```
Ranking Model → Top-20 candidates → [LLM Reranker] → Top-10 final results
```

## Key Features

- **Zero-shot**: No training required, uses pre-trained models
- **Semantic understanding**: Captures meaning beyond embeddings
- **Lightweight**: ~80MB model (all-MiniLM-L6-v2)
- **Optional**: Can be skipped if not needed

## Installation

```bash
# Install transformers library
pip install transformers
```

## Quick Demo

```bash
# Run demo (downloads model on first run)
python models/reranking/demo_reranker.py
```

## Usage

### Simple Reranker (Recommended)

```python
from models.reranking.llm_reranker import SimpleLLMReranker

# Initialize
reranker = SimpleLLMReranker()

# User history (as text)
user_history = [
    "Laptop 15 inch for programming",
    "Wireless mouse gaming",
    "Mechanical keyboard"
]

# Candidates from ranking model
candidates = [
    "USB-C hub for laptop",
    "Gaming headset with mic",
    "Laptop cooling pad",
    "Office chair ergonomic",
    "Monitor 27 inch 4K"
]

# Previous scores from ranking model
previous_scores = [0.65, 0.72, 0.68, 0.45, 0.81]

# Rerank
top_indices, top_scores = reranker.rerank(
    user_history=user_history,
    candidates=candidates,
    candidate_scores=previous_scores,
    top_k=3,
    alpha=0.5  # 50% LLM, 50% ranking model
)

print("Top-3 items:", [candidates[i] for i in top_indices])
```

## Model Options

### Lightweight (Recommended)
- **all-MiniLM-L6-v2**: ~80MB, fast, good quality
- Best for: CPU inference, quick testing

### Better Quality
- **all-mpnet-base-v2**: ~420MB, slower, best quality
- Best for: GPU inference, production

### Usage
```python
# Lightweight
reranker = SimpleLLMReranker("sentence-transformers/all-MiniLM-L6-v2")

# Better quality
reranker = SimpleLLMReranker("sentence-transformers/all-mpnet-base-v2")
```

## Parameters

### alpha (combination weight)
- `alpha=1.0`: LLM only (ignore previous scores)
- `alpha=0.5`: Balanced (50% LLM, 50% ranking model)
- `alpha=0.3`: Conservative (30% LLM, 70% ranking model)

**Recommendation**: Start with `alpha=0.5`, tune based on eval metrics

### top_k
- Number of final items to return
- Typically `top_k=10` for final recommendations

## When to Use LLM Reranker

### ✅ Use When:
1. Have item text descriptions (titles, categories)
2. Want semantic understanding
3. Cold-start problem (new items/users)
4. Cross-category recommendations

### ❌ Skip When:
1. No item text available (only IDs)
2. Latency critical (<10ms required)
3. Very large candidate sets (>100 items)
4. Budget/resource constrained

## Performance

### Latency (all-MiniLM-L6-v2)
- 20 candidates: ~50ms (CPU), ~10ms (GPU)
- 100 candidates: ~200ms (CPU), ~30ms (GPU)

### Memory
- Model: ~80MB
- Per request: ~10MB

### Improvement
- NDCG@10: +5-10% over ranking model alone
- Particularly good for:
  - Cross-category recommendations
  - Semantic similarity
  - Cold-start items

## Integration with API

```python
# In API service
from models.reranking.llm_reranker import SimpleLLMReranker

class RecommendationService:
    def __init__(self):
        # ... load retrieval and ranking models ...
        self.llm_reranker = SimpleLLMReranker()

    def recommend(self, user_id, top_k=10):
        # 1. Retrieval: Get 100 candidates
        candidates = self.retrieval_model.retrieve(user_id, k=100)

        # 2. Ranking: Rank to top-20
        ranked = self.ranking_model.rank(user_id, candidates)
        top_20 = ranked[:20]

        # 3. LLM Reranking: Refine to top-10
        user_history_texts = self.get_user_history_texts(user_id)
        candidate_texts = [self.get_item_text(item) for item in top_20]

        final_indices, final_scores = self.llm_reranker.rerank(
            user_history=user_history_texts,
            candidates=candidate_texts,
            candidate_scores=[item['score'] for item in top_20],
            top_k=top_k,
            alpha=0.5
        )

        return [top_20[i] for i in final_indices]
```

## Limitations

1. **Requires text**: Need item descriptions (not just IDs)
2. **Latency**: Adds 10-200ms depending on setup
3. **Memory**: Requires ~80MB+ for model
4. **Quality**: Depends on text quality

## Alternatives

If LLM reranker doesn't fit:
1. **Skip it**: Use ranking model directly
2. **Simpler approach**: TF-IDF + cosine similarity
3. **Lighter model**: Use smaller transformers

## Notes

- **Optional component**: System works without it
- **Easy to add/remove**: Doesn't affect other models
- **No training**: Uses pre-trained weights
- **Plug-and-play**: Drop into existing pipeline

## Troubleshooting

### "No module named 'transformers'"
```bash
pip install transformers
```

### "Connection error" / Can't download model
- Model downloads on first run (~80MB)
- Requires internet connection
- Downloaded to `~/.cache/huggingface/`

### Slow inference
- Use GPU if available
- Reduce candidate set (top-20 instead of top-100)
- Use lighter model (all-MiniLM-L6-v2)

### Out of memory
- Reduce batch size
- Use CPU instead of GPU
- Use lighter model
