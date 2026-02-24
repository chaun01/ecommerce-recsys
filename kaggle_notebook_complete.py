# ============================================================
# KAGGLE NOTEBOOK - COMPLETE PIPELINE
# Copy từng cell này vào Kaggle Notebook
# ============================================================

# ============================================================
# CELL 1: Setup & Install Dependencies
# ============================================================
!pip install pandas pyarrow pyyaml tqdm scikit-learn -q

import torch
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"✓ CUDA: {torch.cuda.is_available()}")

# ============================================================
# CELL 2: Upload Raw Data to Kaggle Dataset TRƯỚC
# Sau đó add dataset vào notebook (bên phải → Add Input)
# Dataset phải có 4 files:
# - events.csv
# - item_properties_part1.csv
# - item_properties_part2.csv
# - category_tree.csv
# ============================================================

# ============================================================
# CELL 3: Copy raw data từ input dataset
# ============================================================
import shutil
import os

# Thay <YOUR_DATASET_NAME> bằng tên dataset bạn tạo
DATASET_NAME = "retailrocket-raw"  # Thay tên này

os.makedirs('data', exist_ok=True)
shutil.copy(f'/kaggle/input/{DATASET_NAME}/events.csv', 'data/')
shutil.copy(f'/kaggle/input/{DATASET_NAME}/item_properties_part1.csv', 'data/')
shutil.copy(f'/kaggle/input/{DATASET_NAME}/item_properties_part2.csv', 'data/')
shutil.copy(f'/kaggle/input/{DATASET_NAME}/category_tree.csv', 'data/')

print("✓ Raw data copied!")
!ls -lh data/

# ============================================================
# CELL 4: Create preprocessing.py
# ============================================================
%%writefile data/preprocessing.py
