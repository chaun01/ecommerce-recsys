"""
Data preprocessing pipeline for RetailRocket e-commerce dataset.
Converts raw events into user-item interactions with time-based train/val/test split.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")


def load_events(min_user_interactions=5, min_item_interactions=5):
    """Load and filter events data."""
    print("Loading events...")
    df = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Assign implicit feedback weights
    event_weights = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}
    df["weight"] = df["event"].map(event_weights)
    df = df.dropna(subset=["weight"])

    # Filter cold-start users and items
    for _ in range(3):  # iterative filtering
        user_counts = df["visitorid"].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["visitorid"].isin(valid_users)]

        item_counts = df["itemid"].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df["itemid"].isin(valid_items)]

    print(f"  Events after filtering: {len(df):,}")
    print(f"  Users: {df['visitorid'].nunique():,}, Items: {df['itemid'].nunique():,}")
    return df


def load_item_properties():
    """Load and merge item properties (category, etc.)."""
    print("Loading item properties...")
    parts = []
    for fname in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        path = os.path.join(DATA_DIR, fname)
        parts.append(pd.read_csv(path))
    props = pd.concat(parts, ignore_index=True)

    # Extract categoryid — take the latest value per item
    cat_df = props[props["property"] == "categoryid"].copy()
    cat_df = cat_df.sort_values("timestamp").drop_duplicates(subset=["itemid"], keep="last")
    cat_df = cat_df[["itemid", "value"]].rename(columns={"value": "categoryid"})
    cat_df["categoryid"] = pd.to_numeric(cat_df["categoryid"], errors="coerce")
    cat_df = cat_df.dropna(subset=["categoryid"])
    cat_df["categoryid"] = cat_df["categoryid"].astype(int)

    print(f"  Items with category: {len(cat_df):,}")
    return cat_df


def load_category_tree():
    """Load category hierarchy."""
    tree = pd.read_csv(os.path.join(DATA_DIR, "category_tree.csv"))
    return tree


def encode_ids(events_df):
    """Encode user and item IDs to contiguous integers."""
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    events_df["user_idx"] = user_enc.fit_transform(events_df["visitorid"])
    events_df["item_idx"] = item_enc.fit_transform(events_df["itemid"])

    n_users = events_df["user_idx"].nunique()
    n_items = events_df["item_idx"].nunique()
    print(f"  Encoded: {n_users} users, {n_items} items")
    return events_df, user_enc, item_enc


def build_user_histories(events_df):
    """Build chronological interaction history per user."""
    histories = (
        events_df.sort_values("timestamp")
        .groupby("user_idx")
        .apply(lambda g: list(zip(g["item_idx"].values, g["weight"].values, g["timestamp"].values)),
               include_groups=False)
        .to_dict()
    )
    return histories


def time_based_split(events_df, val_ratio=0.1, test_ratio=0.1):
    """Split by timestamp: train / val / test."""
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    n = len(events_df)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))

    train = events_df.iloc[:train_end].copy()
    val = events_df.iloc[train_end:val_end].copy()
    test = events_df.iloc[val_end:].copy()

    print(f"  Split: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return train, val, test


def build_interaction_matrix(events_df, n_users, n_items):
    """Build sparse user-item interaction matrix (aggregated weights)."""
    agg = events_df.groupby(["user_idx", "item_idx"])["weight"].sum().reset_index()
    matrix = csr_matrix(
        (agg["weight"].values, (agg["user_idx"].values, agg["item_idx"].values)),
        shape=(n_users, n_items),
    )
    return matrix


def preprocess_and_save():
    """Run full preprocessing pipeline and save artifacts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and filter events
    events = load_events()

    # 2. Load item metadata
    item_cats = load_item_properties()
    cat_tree = load_category_tree()

    # 3. Encode IDs
    events, user_enc, item_enc = encode_ids(events)
    n_users = events["user_idx"].max() + 1
    n_items = events["item_idx"].max() + 1

    # 4. Merge item categories (for ranking features later)
    item_meta = pd.DataFrame({"itemid": item_enc.classes_, "item_idx": range(n_items)})
    item_meta = item_meta.merge(item_cats, on="itemid", how="left")
    item_meta["categoryid"] = item_meta["categoryid"].fillna(-1).astype(int)

    # Encode categories
    cat_enc = LabelEncoder()
    item_meta["cat_idx"] = cat_enc.fit_transform(item_meta["categoryid"])

    # 5. Time-based split
    train, val, test = time_based_split(events)

    # 6. Build user histories (from train only)
    user_histories = build_user_histories(train)

    # 7. Build interaction matrix (train)
    train_matrix = build_interaction_matrix(train, n_users, n_items)

    # 8. Save everything
    artifacts = {
        "train": train,
        "val": val,
        "test": test,
        "n_users": n_users,
        "n_items": n_items,
        "n_cats": item_meta["cat_idx"].max() + 1,
        "user_histories": user_histories,
        "train_matrix": train_matrix,
        "item_meta": item_meta,
        "cat_tree": cat_tree,
    }

    for name, obj in artifacts.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {name} -> {path}")

    # Save encoders
    encoders = {"user_enc": user_enc, "item_enc": item_enc, "cat_enc": cat_enc}
    with open(os.path.join(OUTPUT_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print("  Saved encoders")

    print("\nPreprocessing complete!")
    return artifacts, encoders


if __name__ == "__main__":
    preprocess_and_save()
