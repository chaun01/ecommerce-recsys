"""
Training script for Wide & Deep ranking model.
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ranking.wide_deep import WideAndDeep
from ranking.dataset import RankingDataset
from retrieval.two_tower import TwoTowerModel
from retrieval.train import compute_user_hist_emb

# ---- Config ----
EMBEDDING_DIM = 64
CAT_EMB_DIM = 16
WIDE_DIM = 32
DEEP_HIDDEN = (128, 64)
BATCH_SIZE = 2048
LR = 1e-3
EPOCHS = 3
N_NEG = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_pkl(name):
    with open(os.path.join(DATA_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def compute_all_user_embeddings(two_tower_model, user_histories, item_meta, n_users, device):
    """Compute user embeddings for all users using the trained two-tower model."""
    two_tower_model.eval()
    user_embeddings = np.zeros((n_users, two_tower_model.embedding_dim), dtype=np.float32)

    with torch.no_grad():
        for uid in tqdm(range(n_users), desc="Computing user embeddings"):
            hist = user_histories.get(uid, [])
            if not hist:
                continue
            hist_ids = [h[0] for h in hist[-50:]]  # last 50 items
            hist_weights = np.array([h[1] for h in hist[-50:]], dtype=np.float32)
            hist_weights = hist_weights / hist_weights.sum()

            hist_ids_t = torch.tensor([hist_ids], dtype=torch.long).to(device)
            hist_weights_t = torch.tensor([hist_weights], dtype=torch.float32).to(device)

            user_hist_emb = compute_user_hist_emb(two_tower_model, hist_ids_t, hist_weights_t, device)
            user_emb = two_tower_model.get_user_embeddings(user_hist_emb)
            user_emb = torch.nn.functional.normalize(user_emb, dim=-1)
            user_embeddings[uid] = user_emb.cpu().numpy()[0]

    return user_embeddings


def train_epoch(model, dataloader, optimizer, user_embeddings, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        B = batch["item_ids"].shape[0]
        n_cand = batch["item_ids"].shape[1]  # 1 + n_neg

        # Get user embeddings
        user_embs = user_embeddings[batch["user_idx"]]  # (B, D)
        user_embs = torch.tensor(user_embs, dtype=torch.float32).to(device)
        user_embs = user_embs.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)

        # Flatten candidates
        item_ids = batch["item_ids"].reshape(-1).to(device)
        cat_ids = batch["cat_ids"].reshape(-1).to(device)
        ret_scores = batch["retrieval_scores"].reshape(-1).to(device)
        evt_counts = batch["event_counts"].reshape(-1).to(device)
        recency = batch["recency"].reshape(-1).to(device)
        labels = batch["labels"].reshape(-1).to(device)

        scores = model(user_embs, item_ids, cat_ids, ret_scores, evt_counts, recency)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, user_embeddings, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            B = batch["item_ids"].shape[0]
            n_cand = batch["item_ids"].shape[1]

            user_embs = user_embeddings[batch["user_idx"]]
            user_embs = torch.tensor(user_embs, dtype=torch.float32).to(device)
            user_embs = user_embs.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)

            item_ids = batch["item_ids"].reshape(-1).to(device)
            cat_ids = batch["cat_ids"].reshape(-1).to(device)
            ret_scores = batch["retrieval_scores"].reshape(-1).to(device)
            evt_counts = batch["event_counts"].reshape(-1).to(device)
            recency = batch["recency"].reshape(-1).to(device)
            labels = batch["labels"].reshape(-1).to(device)

            scores = model(user_embs, item_ids, cat_ids, ret_scores, evt_counts, recency)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print(f"Device: {DEVICE}")

    # Load data
    train_df = load_pkl("train")
    val_df = load_pkl("val")
    n_users = load_pkl("n_users")
    n_items = load_pkl("n_items")
    n_cats = load_pkl("n_cats")
    user_histories = load_pkl("user_histories")
    item_meta = load_pkl("item_meta")

    # Load trained two-tower model
    with open(os.path.join(MODEL_DIR, "two_tower_config.pkl"), "rb") as f:
        tt_config = pickle.load(f)

    two_tower = TwoTowerModel(**tt_config).to(DEVICE)
    two_tower.load_state_dict(torch.load(os.path.join(MODEL_DIR, "two_tower.pt"), weights_only=True))

    # Compute user embeddings
    user_emb_path = os.path.join(MODEL_DIR, "user_embeddings.npy")
    if os.path.exists(user_emb_path):
        print("Loading cached user embeddings...")
        user_embeddings = np.load(user_emb_path)
    else:
        user_embeddings = compute_all_user_embeddings(
            two_tower, user_histories, item_meta, n_users, DEVICE
        )
        np.save(user_emb_path, user_embeddings)
        print(f"Saved user embeddings: {user_embeddings.shape}")

    item_embeddings = np.load(os.path.join(MODEL_DIR, "item_embeddings.npy"))

    # Datasets
    train_ds = RankingDataset(train_df, user_histories, item_meta,
                               user_embeddings, item_embeddings, n_items, N_NEG)
    val_ds = RankingDataset(val_df, user_histories, item_meta,
                             user_embeddings, item_embeddings, n_items, N_NEG)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = WideAndDeep(
        n_items=n_items, n_cats=n_cats,
        embedding_dim=EMBEDDING_DIM, cat_emb_dim=CAT_EMB_DIM,
        wide_dim=WIDE_DIM, deep_hidden_dims=DEEP_HIDDEN,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, user_embeddings, DEVICE)
        val_loss = validate(model, val_loader, user_embeddings, DEVICE)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "wide_deep.pt"))
            print("  -> Saved best model")

    # Save model config
    config = {
        "n_items": n_items,
        "n_cats": n_cats,
        "embedding_dim": EMBEDDING_DIM,
        "cat_emb_dim": CAT_EMB_DIM,
        "wide_dim": WIDE_DIM,
        "deep_hidden_dims": DEEP_HIDDEN,
    }
    with open(os.path.join(MODEL_DIR, "wide_deep_config.pkl"), "wb") as f:
        pickle.dump(config, f)

    print("Ranking model training complete!")


if __name__ == "__main__":
    main()
