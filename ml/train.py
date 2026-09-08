"""
Training script for OceanEmbedModel.

Usage:
    python -m ml.train
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from ml import config
from ml.dataset import get_dataloader
from ml.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(mode=train)
    total_loss = 0.0
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            if train:
                loss.backward()
                optimizer.step()
            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            n_samples += batch_size

    return total_loss / max(n_samples, 1)


def plot_learning_curve(train_losses, val_losses, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (normalized targets)")
    plt.title("OceanEmbed Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved learning curve -> {out_path}")


def train() -> None:
    print("=" * 70)
    print("OceanEmbed - Training")
    print("=" * 70)

    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Input variables: {config.INPUT_VARIABLES}")
    print(f"Batch size: {config.BATCH_SIZE}, LR: {config.LEARNING_RATE}, "
          f"Epochs: {config.NUM_EPOCHS}, Embedding dim: {config.EMBEDDING_DIM}")

    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses, val_losses = [], []

    history_path = config.METRICS_DIR / "training_history.json"

    print("\nStarting training...\n")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}   ({elapsed:.1f}s)")

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": {
                        "input_variables": config.INPUT_VARIABLES,
                        "conv_channels": config.CONV_CHANNELS,
                        "embedding_dim": config.EMBEDDING_DIM,
                        "decoder_hidden_dims": config.DECODER_HIDDEN_DIMS,
                        "n_depths": config.N_DEPTHS,
                        "patch_size": config.PATCH_SIZE,
                    },
                },
                config.BEST_CHECKPOINT_PATH,
            )
            print(f"  -> New best model saved to {config.BEST_CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs "
                  f"(no improvement for {config.EARLY_STOPPING_PATIENCE} epochs).")
            break

    # Always also save the final-epoch model (even if not the best).
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": len(train_losses),
            "val_loss": val_losses[-1] if val_losses else None,
            "config": {
                "input_variables": config.INPUT_VARIABLES,
                "conv_channels": config.CONV_CHANNELS,
                "embedding_dim": config.EMBEDDING_DIM,
                "decoder_hidden_dims": config.DECODER_HIDDEN_DIMS,
                "n_depths": config.N_DEPTHS,
                "patch_size": config.PATCH_SIZE,
            },
        },
        config.FINAL_CHECKPOINT_PATH,
    )
    print(f"\nSaved final-epoch model -> {config.FINAL_CHECKPOINT_PATH}")

    with open(history_path, "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=2)
    print(f"Saved training history -> {history_path}")

    plot_learning_curve(train_losses, val_losses, config.PLOTS_DIR / "learning_curve.png")

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print("Training complete.")
    print("=" * 70)


if __name__ == "__main__":
    train()
