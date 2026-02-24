# Two-Tower Retrieval Model

This module implements a Two-Tower neural network for candidate retrieval in the recommendation system.

## Architecture

### User Tower
- **Input**: Sequence of item IDs (user's interaction history)
- **Embedding**: Item embeddings
- **Pooling**: Mean/Max/Sum pooling over sequence
- **MLP**: Multi-layer perceptron to produce user embedding
- **Output**: L2-normalized user embedding (64-dim)

### Item Tower
- **Input**: Item ID + Category ID
- **Embedding**: Item and category embeddings
- **MLP**: Multi-layer perceptron to produce item embedding
- **Output**: L2-normalized item embedding (64-dim)

### Training
- **Loss**: InfoNCE (contrastive loss)
- **Positive samples**: Items user actually interacted with
- **Negative samples**:
  - In-batch negatives (all other items in the batch)
  - Explicit negatives (randomly sampled or popularity-based)

## Files

- `two_tower.py`: Model architecture
- `dataset.py`: Dataset and DataLoader
- `train_retrieval.py`: Training script

## Training

### Quick Start

```bash
# Train with default config
python models/retrieval/train_retrieval.py --config configs/retrieval_config.yaml

# Train on CPU
python models/retrieval/train_retrieval.py --device cpu

# Train on GPU
python models/retrieval/train_retrieval.py --device cuda
```

### Configuration

Edit `configs/retrieval_config.yaml` to customize:
- Model architecture (embedding_dim, hidden_dims, dropout)
- Training hyperparameters (batch_size, learning_rate, epochs)
- Data settings (max_sequence_length, num_negatives)

### Key Hyperparameters

- `embedding_dim`: Dimension of embeddings (default: 64)
- `temperature`: Temperature for contrastive loss (default: 0.07)
- `num_negatives`: Number of explicit negative samples (default: 4)
- `batch_size`: Batch size (default: 512)
- `learning_rate`: Learning rate (default: 0.001)

## Output

After training, the following files will be saved to `models/retrieval/checkpoints/`:

- `best_model.pth`: Best model checkpoint (lowest validation loss)
- `checkpoint_epoch_*.pth`: Checkpoints for each epoch
- `user_embeddings.npy`: User embeddings (shape: [num_users, embedding_dim])
- `item_embeddings.npy`: Item embeddings (shape: [num_items, embedding_dim])
- `training_history.csv`: Training metrics over time

## Usage in Inference

```python
import torch
import numpy as np
from models.retrieval.two_tower import TwoTowerModel

# Load model
checkpoint = torch.load('models/retrieval/checkpoints/best_model.pth')
model = create_two_tower_model(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load embeddings
item_embeddings = np.load('models/retrieval/checkpoints/item_embeddings.npy')

# Encode a new user
user_history = torch.LongTensor([[101, 203, 405, 0, 0]])  # padded sequence
user_embedding = model.encode_user(user_history)

# Compute similarity scores
scores = torch.matmul(user_embedding, torch.FloatTensor(item_embeddings).T)
top_k_items = scores.topk(100).indices  # Top 100 candidates
```

## Model Size

- Parameters: ~5-10M (depending on configuration)
- User embeddings: ~10-20 MB (for 180K users, 64-dim)
- Item embeddings: ~5 MB (for 68K items, 64-dim)

## Training Time

- 1 epoch: ~3-5 minutes on GPU (RTX 3090)
- Full training (20 epochs): ~1-2 hours

## Expected Performance

- Validation loss: ~2.5-3.0 (after 20 epochs)
- Recall@10: ~0.10-0.15 (retrieval only)
- Recall@100: ~0.30-0.40
