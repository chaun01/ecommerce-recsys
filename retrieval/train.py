"""
Training script for Two-Tower retrieval model.
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.two_tower import TwoTowerModel
from retrieval.dataset import TwoTowerDataset, collate_fn

# ---- Config ----
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
CAT_EMBEDDING_DIM = 16
BATCH_SIZE = 2048
LR = 1e-3
EPOCHS = 3
N_NEG = 4
MAX_HIST_LEN = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data():
    def load_pkl(name):
        with open(os.path.join(DATA_DIR, f"{name}.pkl"), "rb") as f:
            return pickle.load(f)

    return {
        "train": load_pkl("train"),
        "val": load_pkl("val"),
        "n_users": load_pkl("n_users"),
        "n_items": load_pkl("n_items"),
        "n_cats": load_pkl("n_cats"),
        "user_histories": load_pkl("user_histories"),
        "item_meta": load_pkl("item_meta"),
    }


def compute_user_hist_emb(model, hist_ids, hist_weights, device):
    """Compute weighted average of item embeddings for user history."""
    # hist_ids: (B, L), hist_weights: (B, L)
    item_embs = model.item_tower.item_emb(hist_ids.to(device))  # (B, L, D)
    weights = hist_weights.to(device).unsqueeze(-1)               # (B, L, 1)
    user_hist_emb = (item_embs * weights).sum(dim=1)              # (B, D)
    return user_hist_emb


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        user_hist_emb = compute_user_hist_emb(
            model, batch["hist_ids"], batch["hist_weights"], device
        )
        loss = model(
            user_hist_emb,
            batch["pos_item"].to(device),
            batch["pos_cat"].to(device),
            batch["neg_items"].to(device),
            batch["neg_cats"].to(device),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            user_hist_emb = compute_user_hist_emb(
                model, batch["hist_ids"], batch["hist_weights"], device
            )
            loss = model(
                user_hist_emb,
                batch["pos_item"].to(device),
                batch["pos_cat"].to(device),
                batch["neg_items"].to(device),
                batch["neg_cats"].to(device),
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


def export_embeddings(model, item_meta, device):
    """Export all item embeddings for FAISS indexing."""
    model.eval()
    all_item_ids = torch.arange(len(item_meta), dtype=torch.long)
    all_cat_ids = torch.tensor(item_meta["cat_idx"].values, dtype=torch.long)

    with torch.no_grad():
        # Process in batches to avoid OOM
        embeddings = []
        bs = 4096
        for i in range(0, len(all_item_ids), bs):
            item_batch = all_item_ids[i:i+bs].to(device)
            cat_batch = all_cat_ids[i:i+bs].to(device)
            emb = model.get_item_embeddings(item_batch, cat_batch)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            embeddings.append(emb.cpu().numpy())

    item_embeddings = np.concatenate(embeddings, axis=0)
    return item_embeddings


def main():
    print(f"Device: {DEVICE}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    data = load_data()
    print(f"Users: {data['n_users']}, Items: {data['n_items']}, Categories: {data['n_cats']}")

    # Datasets
    train_ds = TwoTowerDataset(
        data["train"], data["user_histories"], data["item_meta"],
        data["n_items"], MAX_HIST_LEN, N_NEG
    )
    val_ds = TwoTowerDataset(
        data["val"], data["user_histories"], data["item_meta"],
        data["n_items"], MAX_HIST_LEN, N_NEG
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Model
    model = TwoTowerModel(
        n_items=data["n_items"],
        n_cats=data["n_cats"],
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        cat_embedding_dim=CAT_EMBEDDING_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate(model, val_loader, DEVICE)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "two_tower.pt"))
            print("  -> Saved best model")

    # Load best and export embeddings
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "two_tower.pt"), weights_only=True))
    item_embeddings = export_embeddings(model, data["item_meta"], DEVICE)
    np.save(os.path.join(MODEL_DIR, "item_embeddings.npy"), item_embeddings)
    print(f"Exported item embeddings: {item_embeddings.shape}")

    # Save model config for loading later
    config = {
        "n_items": data["n_items"],
        "n_cats": data["n_cats"],
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "cat_embedding_dim": CAT_EMBEDDING_DIM,
    }
    with open(os.path.join(MODEL_DIR, "two_tower_config.pkl"), "wb") as f:
        pickle.dump(config, f)

    print("Training complete!")


if __name__ == "__main__":
    main()
