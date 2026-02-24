"""
Training script for Two-Tower retrieval model.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.retrieval.two_tower import TwoTowerModel, create_two_tower_model
from models.retrieval.dataset import create_dataloaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalTrainer:
    """Trainer for Two-Tower retrieval model."""

    def __init__(
        self,
        model: TwoTowerModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize trainer.

        Args:
            model: Two-Tower model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training config
        training_config = config["training"]
        self.epochs = training_config["epochs"]
        self.learning_rate = training_config["learning_rate"]
        self.weight_decay = training_config.get("weight_decay", 0.0001)
        self.log_interval = config["logging"].get("log_interval", 100)
        self.save_interval = config["logging"].get("save_interval", 1)

        # Output directory
        self.output_dir = Path(config["logging"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        optimizer_type = training_config.get("optimizer", "adamw").lower()
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Learning rate scheduler
        scheduler_config = training_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine")

        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 5),
                gamma=scheduler_config.get("gamma", 0.5)
            )
        else:
            self.scheduler = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }

        self.best_val_loss = float('inf')
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                user_item_sequence=batch['user_item_sequence'],
                positive_items=batch['positive_items'],
                negative_items=batch['negative_items'],
                user_sequence_lengths=batch['sequence_length'],
                positive_categories=batch['positive_categories'],
                negative_categories=batch['negative_categories']
            )

            loss = outputs['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / num_batches
            })

            # Log
            if self.global_step % self.log_interval == 0:
                logger.info(
                    f"Step {self.global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {total_loss / num_batches:.4f}"
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.val_loader, desc="Validation")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                user_item_sequence=batch['user_item_sequence'],
                positive_items=batch['positive_items'],
                negative_items=batch['negative_items'],
                user_sequence_lengths=batch['sequence_length'],
                positive_categories=batch['positive_categories'],
                negative_categories=batch['negative_categories']
            )

            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def save_embeddings(self):
        """Save user and item embeddings."""
        logger.info("Generating embeddings...")
        self.model.eval()

        # Generate item embeddings
        item_features = pd.read_parquet(self.config["data"]["item_features_file"])
        item_ids = torch.LongTensor(item_features['item_idx'].values).to(self.device)
        category_ids = torch.LongTensor(item_features['categoryid'].values).to(self.device)

        item_embeddings = []
        batch_size = 1024

        with torch.no_grad():
            for i in tqdm(range(0, len(item_ids), batch_size), desc="Item embeddings"):
                batch_items = item_ids[i:i+batch_size]
                batch_cats = category_ids[i:i+batch_size]
                embeds = self.model.encode_item(batch_items, batch_cats)
                item_embeddings.append(embeds.cpu().numpy())

        item_embeddings = np.vstack(item_embeddings)

        # Save item embeddings
        item_embed_path = self.output_dir / "item_embeddings.npy"
        np.save(item_embed_path, item_embeddings)
        logger.info(f"Saved item embeddings to {item_embed_path} with shape {item_embeddings.shape}")

        # Generate user embeddings from sequences
        sequences = pd.read_parquet(self.config["data"]["train_file"].replace("interactions", "sequences"))

        user_embeddings = []
        max_seq_len = self.config["data"]["max_sequence_length"]

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="User embeddings"):
                batch_sequences = sequences.iloc[i:i+batch_size]

                # Pad sequences
                padded_seqs = []
                seq_lengths = []

                for _, row in batch_sequences.iterrows():
                    seq = row['item_sequence']
                    if len(seq) < max_seq_len:
                        padded = np.pad(seq, (0, max_seq_len - len(seq)), constant_values=0)
                        seq_len = len(seq)
                    else:
                        padded = seq[-max_seq_len:]
                        seq_len = max_seq_len

                    padded_seqs.append(padded)
                    seq_lengths.append(seq_len)

                batch_tensor = torch.LongTensor(np.array(padded_seqs)).to(self.device)
                seq_lens = torch.LongTensor(seq_lengths).to(self.device)

                embeds = self.model.encode_user(batch_tensor, seq_lens)
                user_embeddings.append(embeds.cpu().numpy())

        user_embeddings = np.vstack(user_embeddings)

        # Save user embeddings
        user_embed_path = self.output_dir / "user_embeddings.npy"
        np.save(user_embed_path, user_embeddings)
        logger.info(f"Saved user embeddings to {user_embed_path} with shape {user_embeddings.shape}")

    def train(self):
        """Run training loop."""
        logger.info("="*80)
        logger.info("Starting Two-Tower Retrieval Model Training")
        logger.info("="*80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info("="*80)

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss = self.validate()
            self.history["val_loss"].append(val_loss)

            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["learning_rate"].append(current_lr)

            # Log epoch results
            logger.info("="*80)
            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            logger.info("="*80)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch, is_best=is_best)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

        # Save final embeddings
        self.save_embeddings()

        # Save training history
        history_df = pd.DataFrame(self.history)
        history_path = self.output_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info(f"Saved training history to {history_path}")

        logger.info("="*80)
        logger.info("Training Complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("="*80)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Two-Tower retrieval model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval_config.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset statistics
    stats_path = Path(config["data"]["train_file"]).parent / "dataset_stats.parquet"
    stats = pd.read_parquet(stats_path).iloc[0]

    # Get number of items and categories from item_features (not stats)
    item_features = pd.read_parquet(config["data"]["item_features_file"])
    # num_items must be max_item_idx + 1 to include all indices from 0 to max_item_idx
    config["data"]["num_items"] = int(item_features['item_idx'].max()) + 1
    config["data"]["num_categories"] = int(item_features['categoryid'].max()) + 1

    logger.info(f"Number of items: {config['data']['num_items']}")
    logger.info(f"Number of categories: {config['data']['num_categories']}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_interactions_path=config["data"]["train_file"],
        val_interactions_path=config["data"]["val_file"],
        train_sequences_path=config["data"]["train_file"].replace("interactions", "sequences"),
        item_features_path=config["data"]["item_features_file"],
        batch_size=config["training"]["batch_size"],
        num_workers=4,
        max_sequence_length=config["data"]["max_sequence_length"],
        num_negatives=config["training"]["num_negatives"],
        negative_sampling=config["training"].get("negative_sampling", "random")
    )

    # Create model
    model = create_two_tower_model(config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = RetrievalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
