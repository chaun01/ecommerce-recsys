# Hướng Dẫn Chạy Trên Kaggle - Từng Bước

## BƯỚC 1: Upload Raw Data lên Kaggle Dataset

1. Vào https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Kéo thả 4 files từ folder `c:\Users\Luan\Downloads\chau\recsys\data\`:
   - `events.csv`
   - `item_properties_part1.csv`
   - `item_properties_part2.csv`
   - `category_tree.csv`
4. Đặt tên: **`retailrocket-raw`**
5. Click **"Create"**

---

## BƯỚC 2: Tạo Kaggle Notebook

1. Vào https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Click **"Settings"** (góc phải) → chọn **"Accelerator: GPU P100"**
4. Click **"Save"**

---

## BƯỚC 3: Add Dataset vào Notebook

1. Bên phải màn hình, panel **"Input"**
2. Click **"+ Add Input"**
3. Tìm dataset **"retailrocket-raw"** vừa tạo
4. Click **"Add"**

---

## BƯỚC 4: Upload Code Files

Có 2 cách:

### Cách A: Upload qua GitHub (Nhanh nhất - Recommend)

```python
# CELL 1: Clone repo
!git clone https://github.com/YOUR_USERNAME/recsys.git
%cd recsys
```

Bạn cần push code lên GitHub trước. Tôi có thể giúp tạo file `.gitignore` và hướng dẫn push.

### Cách B: Upload từng file thủ công

Tôi sẽ tạo 1 file Python duy nhất chứa toàn bộ code để bạn copy-paste.

---

## Bạn muốn dùng cách nào?

**A) Push lên GitHub rồi clone** (Nhanh, professional)
**B) Copy-paste code trực tiếp** (Đơn giản, không cần GitHub)

Chọn A hay B?
