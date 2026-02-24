# Chạy TOÀN BỘ trên Kaggle (Preprocessing + Training)

## Bước 1: Upload raw data lên Kaggle Dataset

1. Vào https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload 3 files từ folder `data/`:
   - `events.csv`
   - `item_properties_part1.csv`
   - `item_properties_part2.csv`
   - `category_tree.csv`
4. Đặt tên: `retailrocket-raw`

## Bước 2: Chuẩn bị code

1. **Zip các folders này từ máy bạn:**
   ```bash
   # Trên Windows, chọn các folders sau → Right click → Send to → Compressed folder
   - models/
   - configs/
   - data/preprocessing.py
   ```

   Hoặc dùng command:
   ```bash
   # Tạo file recsys-code.zip chứa models/, configs/, data/
   ```

2. **Upload vào Kaggle Notebook:**
   - Mở Kaggle Notebook
   - Bên phải màn hình → Click **"+ Add Data"**
   - Chọn tab **"Upload"** (không phải "Your Datasets")
   - Click **"Select Files"** → chọn `recsys-code.zip`
   - Đợi upload xong (file sẽ ở `/kaggle/input/recsys-code/`)

## Bước 3: Tạo Kaggle Notebook với GPU

1. Vào https://www.kaggle.com/code
2. New Notebook
3. Settings → **GPU P100**
4. Add Dataset: `retailrocket-raw`

## Bước 4: Chạy code trong notebook

```python
# ========== SETUP ==========
!pip install pandas pyarrow pyyaml tqdm scikit-learn

import os
import shutil
import zipfile

# ========== UPLOAD CODE ==========
# Click "Add data" → "Upload" → chọn file zip chứa models/, configs/, data/
# Hoặc clone từ GitHub nếu bạn đã push code lên

# Nếu upload zip:
!unzip /kaggle/input/recsys-code/recsys-code.zip -d /kaggle/working/

# ========== COPY RAW DATA ==========
os.makedirs('data', exist_ok=True)
shutil.copy('/kaggle/input/retailrocket-raw/events.csv', 'data/')
shutil.copy('/kaggle/input/retailrocket-raw/item_properties_part1.csv', 'data/')
shutil.copy('/kaggle/input/retailrocket-raw/item_properties_part2.csv', 'data/')
shutil.copy('/kaggle/input/retailrocket-raw/category_tree.csv', 'data/')

# ========== CHECK GPU ==========
import torch
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")

# ========== RUN PREPROCESSING (Kaggle có 16GB RAM, đủ để xử lý) ==========
!python data/preprocessing.py --input_dir data --output_dir data/processed --min_interactions 3

# Kiểm tra output
!ls -lh data/processed/

# ========== TRAIN RETRIEVAL MODEL (GPU P100) ==========
# Sửa epochs về 20 để train full
!python models/retrieval/train_retrieval.py --config configs/retrieval_config.yaml --device cuda

# ========== DOWNLOAD RESULTS ==========
# Zip checkpoints
!zip -r model_outputs.zip models/retrieval/checkpoints/ data/processed/

# Click Download button để tải về máy
```

## Bước 5: Download về máy

Sau khi chạy xong, download file `model_outputs.zip` và extract vào project.

---

## Thời gian ước tính:

- **Preprocessing**: ~5-10 phút (Kaggle có 16GB RAM)
- **Training (20 epochs GPU P100)**: ~10-15 phút
- **Tổng**: ~20-25 phút

## Lợi ích:

✓ Không cần RAM trên máy local
✓ Có GPU miễn phí
✓ Làm toàn bộ 1 lần trên cloud
✓ Không tốn điện máy bạn
