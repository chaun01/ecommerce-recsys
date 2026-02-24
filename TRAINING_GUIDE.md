# 🚀 Complete Training Guide

## 📋 Bước 1: Kiểm Tra Data

```bash
# Kiểm tra data đã có chưa
ls data/processed/

# Cần có các files:
# ✓ train_interactions.parquet (857,810 rows)
# ✓ val_interactions.parquet (183,816 rows)
# ✓ test_interactions.parquet (183,817 rows)
# ✓ train_sequences.parquet (130,735 users)
# ✓ item_features.parquet (68,067 items)
```

### Nếu chưa có → Chạy preprocessing:

```bash
python data/preprocessing.py \
  --input_dir data \
  --output_dir data/processed \
  --min_interactions 3
```

**Thời gian**: ~5-10 phút
**Output**: Files .parquet trong `data/processed/`

---

## 🎯 Bước 2: Train Retrieval Model

### Option A: Train Đầy Đủ (Recommended)

```bash
python models/retrieval/train_retrieval.py \
  --config configs/retrieval_config.yaml \
  --device cpu
```

**Thông số hiện tại**:
- Epochs: 5 (giảm từ 20 để nhanh hơn)
- Batch size: 512
- Embedding dim: 64
- Device: CPU

**Thời gian ước tính**:
- CPU: ~30-60 phút (5 epochs)
- GPU: ~10-15 phút (5 epochs)

### Option B: Train Nhanh (Quick Test)

Sửa config để train cực nhanh:

```yaml
# configs/retrieval_config.yaml
training:
  batch_size: 512
  epochs: 2  # Chỉ 2 epochs
  learning_rate: 0.001
```

```bash
python models/retrieval/train_retrieval.py \
  --config configs/retrieval_config.yaml \
  --device cpu
```

**Thời gian**: ~15-20 phút

### Option C: Debug Mode (Kiểm tra code chạy)

Tạo config test nhỏ:

```yaml
# configs/retrieval_config_test.yaml
training:
  batch_size: 64   # Nhỏ
  epochs: 1        # 1 epoch thôi
```

```bash
python models/retrieval/train_retrieval.py \
  --config configs/retrieval_config_test.yaml \
  --device cpu
```

**Thời gian**: ~5 phút

---

## 📊 Output Của Training

Sau khi train xong, sẽ có:

```
models/retrieval/checkpoints/
├── best_model.pth              # Model weights
├── checkpoint_epoch_1.pth      # Checkpoint epoch 1
├── checkpoint_epoch_2.pth      # Checkpoint epoch 2
├── ...
├── user_embeddings.npy         # User embeddings (180K × 64)
├── item_embeddings.npy         # Item embeddings (68K × 64)
└── training_history.csv        # Loss/metrics over time
```

---

## 🔍 Monitor Training

### Xem logs trong khi train:

```bash
# Terminal 1: Train
python models/retrieval/train_retrieval.py ...

# Terminal 2: Monitor
tail -f models/retrieval/checkpoints/*.log  # Nếu có
```

### Training output sẽ show:

```
================================================================================
Starting Two-Tower Retrieval Model Training
================================================================================
Device: cpu
Epochs: 5
Learning rate: 0.001
Train batches: 1,670
Val batches: 359
================================================================================

Epoch 1/5:
100%|████████| 1670/1670 [05:23<00:00, 5.17it/s, loss=3.2145, avg_loss=3.4521]

Validation:
100%|████████| 359/359 [00:45<00:00, 7.91it/s, loss=3.1234]

================================================================================
Epoch 1/5
Train Loss: 3.4521
Val Loss: 3.1234
Learning Rate: 0.001000
================================================================================

Saved checkpoint to models/retrieval/checkpoints/checkpoint_epoch_1.pth
```

---

## ⚠️ Troubleshooting

### Lỗi: "No module named 'tqdm'"

```bash
pip install tqdm
```

### Lỗi: "CUDA out of memory"

```bash
# Giảm batch size trong config
batch_size: 256  # Hoặc 128
```

### Lỗi: "FileNotFoundError: train_interactions.parquet"

```bash
# Chạy preprocessing trước
python data/preprocessing.py
```

### Training quá chậm

**Giải pháp**:
1. Giảm epochs: `epochs: 2`
2. Giảm batch_size: `batch_size: 256`
3. Sample data nhỏ hơn trong preprocessing

### Muốn pause training

- `Ctrl+C` để dừng
- Training sẽ save checkpoint mỗi epoch
- Có thể resume từ checkpoint (cần implement resume logic)

---

## ✅ Verify Training Thành Công

```bash
# Kiểm tra files đã tạo
ls -lh models/retrieval/checkpoints/

# Kiểm tra embeddings
python -c "
import numpy as np
user_emb = np.load('models/retrieval/checkpoints/user_embeddings.npy')
item_emb = np.load('models/retrieval/checkpoints/item_embeddings.npy')
print(f'User embeddings: {user_emb.shape}')
print(f'Item embeddings: {item_emb.shape}')
"
```

**Expected output**:
```
User embeddings: (130735, 64)
Item embeddings: (68067, 64)
```

---

## 🎯 Bước Tiếp Theo Sau Training

### 1. Build FAISS Index (~2 phút)

```bash
python retrieval/build_index.py \
  --embeddings models/retrieval/checkpoints/item_embeddings.npy \
  --output retrieval/indices/item_index.faiss \
  --index_type HNSW \
  --benchmark
```

### 2. Test Retrieval

```python
# test_retrieval.py
import numpy as np
from retrieval.faiss_index import FAISSIndex

# Load index
index = FAISSIndex.load('retrieval/indices/item_index.faiss')

# Load user embeddings
user_emb = np.load('models/retrieval/checkpoints/user_embeddings.npy')

# Test retrieval for user 0
user_0_emb = user_emb[0:1]
top_items = index.search(user_0_emb, k=10)

print(f"Top 10 items for user 0: {top_items}")
```

### 3. Run Evaluation

```bash
python evaluation/evaluate.py \
  --test_data data/processed/test_interactions.parquet
```

---

## 📈 Expected Performance

### Training Loss

- **Epoch 1**: Loss ~4.0-5.0
- **Epoch 5**: Loss ~2.5-3.5
- **Val Loss**: Should be close to train loss

### Metrics (After Full Training)

- **Recall@10**: ~0.10-0.15
- **Recall@100**: ~0.30-0.40
- **NDCG@10**: ~0.08-0.12

*Note: Chỉ 5 epochs nên performance sẽ thấp hơn 20 epochs*

---

## 💡 Tips

### Train Nhanh Nhất

```yaml
# Minimal config
training:
  batch_size: 1024  # Lớn nhất có thể
  epochs: 2
  num_negatives: 2  # Giảm negatives

data:
  max_sequence_length: 20  # Giảm sequence length
```

### Train Tốt Nhất

```yaml
# Full config
training:
  batch_size: 512
  epochs: 20
  num_negatives: 4

data:
  max_sequence_length: 50
```

### Debug Training

```python
# Test một batch
python -c "
from models.retrieval.dataset import create_dataloaders

train_loader, val_loader = create_dataloaders(
    'data/processed/train_interactions.parquet',
    'data/processed/val_interactions.parquet',
    'data/processed/train_sequences.parquet',
    'data/processed/item_features.parquet',
    batch_size=64,
    num_workers=0  # 0 for debugging
)

# Get one batch
batch = next(iter(train_loader))
print('Batch keys:', batch.keys())
print('Batch shapes:', {k: v.shape for k, v in batch.items()})
"
```

---

## 🚀 Quick Start Command

**Chạy ngay lệnh này để train:**

```bash
cd "c:\Users\Luan\Downloads\chau\recsys"
python models/retrieval/train_retrieval.py --config configs/retrieval_config.yaml --device cpu
```

**Đợi ~30-60 phút**, sau đó check:

```bash
ls models/retrieval/checkpoints/
```

Xong! 🎉

---

## ❓ FAQ

**Q: Bao lâu mới train xong?**
A: 5 epochs ~30-60 phút (CPU), ~10-15 phút (GPU)

**Q: Có thể train trên Colab không?**
A: Có! Upload code + data lên, chạy với `--device cuda`

**Q: Loss không giảm?**
A: Bình thường nếu chỉ 2-5 epochs. Cần 10-20 epochs mới tốt.

**Q: Có cần train ranking model không?**
A: KHÔNG bắt buộc. Retrieval model + FAISS đã đủ demo.

**Q: Có thể skip training không?**
A: Mock embeddings OK cho demo API, nhưng metrics sẽ random.

---

## 📞 Need Help?

Nếu gặp lỗi:
1. Check logs
2. Verify data files exist
3. Try smaller batch_size
4. Use fewer epochs (2-3)
5. Ask for help!
