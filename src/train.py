"""
Training script for the Flood Prediction LSTM model.

Implements:
- Training loop with early stopping
- Validation monitoring
- Learning rate scheduling
- Model checkpointing
- Training history logging
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import FloodLSTM
from dataset import load_and_split


def train_model(
    csv_path: str,
    output_dir: str = "../results",
    seq_len: int = 30,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 15,
    device: str = None,
    seed: int = 42,
):
    """
    Train the FloodLSTM model and save results.

    Parameters
    ----------
    csv_path : str
        Path to the hydrological CSV.
    output_dir : str
        Directory for saving model weights, history, and scalers.
    seq_len : int
        Input sequence length (days).
    hidden_size : int
        LSTM hidden units.
    num_layers : int
        LSTM layers.
    dropout : float
        Dropout rate.
    lr : float
        Initial learning rate.
    batch_size : int
        Training batch size.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    device : str
        'cuda' or 'cpu'. Auto-detected if None.
    seed : int
        Random seed, matching the default in generate_data.py. Two things here draw on
        the global RNG: LSTM weight initialisation, and the shuffling of the training
        DataLoader. Without a seed the same command produces a different model each run,
        so the reported RMSE could not be reproduced from the repository alone.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Seed before anything constructs a tensor or a loader.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    os.makedirs(output_dir, exist_ok=True)
    scaler_dir = os.path.join(output_dir, "scalers")

    # Load data
    train_loader, val_loader, test_loader, info = load_and_split(
        csv_path, seq_len=seq_len, batch_size=batch_size, scaler_dir=scaler_dir
    )

    n_features = len(info["feature_cols"])

    # Model
    model = FloodLSTM(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Sequence length: {seq_len} days")
    print(f"Features: {info['feature_cols']}")
    print(f"Target: {info['target_col']}\n")

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        scheduler.step(avg_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.0f}s"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save hyperparameters
    hparams = {
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "total_training_time_s": round(total_time, 1),
        "device": device,
        "n_parameters": model.count_parameters(),
    }
    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    print(f"\nModel and history saved to {output_dir}")
    return model, info


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "synthetic_hydrology.csv"
    )
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    train_model(csv_path=data_path, output_dir=results_dir)
