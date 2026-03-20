"""
Evaluation and visualization for the Flood Prediction LSTM.

Generates:
- Predictions vs actuals time series plot
- Scatter plot (predicted vs observed)
- Training loss curves
- Error distribution histogram
- Flood event detection analysis
- Comprehensive metrics report
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from model import FloodLSTM
from dataset import load_and_split, FEATURE_COLS, TARGET_COL


def evaluate_model(
    csv_path: str,
    results_dir: str = "../results",
    device: str = None,
):
    """
    Load trained model, evaluate on test set, and generate all plots.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load hyperparameters
    with open(os.path.join(results_dir, "hyperparameters.json")) as f:
        hparams = json.load(f)

    seq_len = hparams["seq_len"]

    # Load data
    train_loader, val_loader, test_loader, info = load_and_split(
        csv_path, seq_len=seq_len, batch_size=128
    )

    # Load model
    model = FloodLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=hparams["hidden_size"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(results_dir, "best_model.pt"), map_location=device)
    )
    model.eval()

    # Get predictions on test set
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    preds_scaled = np.concatenate(all_preds)
    targets_scaled = np.concatenate(all_targets)

    # Inverse transform to original scale
    target_scaler = info["target_scaler"]
    preds_original = target_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).flatten()
    targets_original = target_scaler.inverse_transform(
        targets_scaled.reshape(-1, 1)
    ).flatten()
    test_dates = info["test_dates"][: len(preds_original)]

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(targets_original, preds_original))
    mae = mean_absolute_error(targets_original, preds_original)
    r2 = r2_score(targets_original, preds_original)
    nse = 1 - np.sum((targets_original - preds_original) ** 2) / np.sum(
        (targets_original - np.mean(targets_original)) ** 2
    )
    pbias = 100 * np.sum(preds_original - targets_original) / np.sum(targets_original)

    metrics = {
        "RMSE (m³/s)": round(float(rmse), 3),
        "MAE (m³/s)": round(float(mae), 3),
        "R²": round(float(r2), 4),
        "NSE": round(float(nse), 4),
        "PBIAS (%)": round(float(pbias), 3),
        "n_test_samples": len(preds_original),
    }

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("=" * 50)

    # Save metrics
    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Generate Plots ----
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Convert dates for matplotlib
    test_dates_dt = pd.to_datetime(test_dates)

    # 1. Time Series: Predicted vs Observed
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        test_dates_dt,
        targets_original,
        label="Observed",
        alpha=0.8,
        linewidth=0.8,
        color="#2196F3",
    )
    ax.plot(
        test_dates_dt,
        preds_original,
        label="LSTM Predicted",
        alpha=0.8,
        linewidth=0.8,
        color="#FF5722",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title("River Discharge — LSTM Prediction vs Observed")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "timeseries_prediction.png"), dpi=150)
    plt.close(fig)
    print("Saved: timeseries_prediction.png")

    # 2. Zoomed-in view (first 200 days)
    n_zoom = min(200, len(preds_original))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        test_dates_dt[:n_zoom],
        targets_original[:n_zoom],
        label="Observed",
        linewidth=1.2,
        color="#2196F3",
    )
    ax.plot(
        test_dates_dt[:n_zoom],
        preds_original[:n_zoom],
        label="LSTM Predicted",
        linewidth=1.2,
        color="#FF5722",
        linestyle="--",
    )
    ax.fill_between(
        test_dates_dt[:n_zoom],
        targets_original[:n_zoom],
        preds_original[:n_zoom],
        alpha=0.15,
        color="gray",
        label="Error",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title("Zoomed View — First 200 Test Days")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "timeseries_zoomed.png"), dpi=150)
    plt.close(fig)
    print("Saved: timeseries_zoomed.png")

    # 3. Scatter Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets_original, preds_original, alpha=0.3, s=10, c="#673AB7")
    vmin = min(targets_original.min(), preds_original.min())
    vmax = max(targets_original.max(), preds_original.max())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="1:1 Line")
    ax.set_xlabel("Observed Discharge (m³/s)")
    ax.set_ylabel("Predicted Discharge (m³/s)")
    ax.set_title(f"Scatter Plot — R² = {r2:.4f}, NSE = {nse:.4f}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "scatter_plot.png"), dpi=150)
    plt.close(fig)
    print("Saved: scatter_plot.png")

    # 4. Training Loss Curves
    with open(os.path.join(results_dir, "training_history.json")) as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#4CAF50")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#F44336")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["lr"], color="#FF9800")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved: training_curves.png")

    # 5. Error Distribution
    errors = preds_original - targets_original
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(errors, bins=80, color="#009688", edgecolor="white", alpha=0.8)
    ax1.axvline(0, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Prediction Error (m³/s)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Error Distribution — MAE = {mae:.2f} m³/s")

    # Residuals over time
    ax2.scatter(test_dates_dt, errors, alpha=0.2, s=5, c="#795548")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Residual (m³/s)")
    ax2.set_title("Residuals Over Time")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "error_analysis.png"), dpi=150)
    plt.close(fig)
    print("Saved: error_analysis.png")

    # 6. Flood Event Detection
    flood_threshold = np.percentile(targets_original, 95)
    actual_flood = targets_original >= flood_threshold
    predicted_flood = preds_original >= flood_threshold

    tp = np.sum(actual_flood & predicted_flood)
    fp = np.sum(~actual_flood & predicted_flood)
    fn = np.sum(actual_flood & ~predicted_flood)
    tn = np.sum(~actual_flood & ~predicted_flood)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    flood_metrics = {
        "flood_threshold_m3s": round(float(flood_threshold), 2),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }

    print("\nFlood Event Detection (95th percentile threshold):")
    for k, v in flood_metrics.items():
        print(f"  {k:25s}: {v}")

    with open(os.path.join(results_dir, "flood_detection_metrics.json"), "w") as f:
        json.dump(flood_metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        test_dates_dt,
        targets_original,
        alpha=0.6,
        linewidth=0.7,
        color="#2196F3",
        label="Observed",
    )
    ax.axhline(
        flood_threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Flood Threshold ({flood_threshold:.1f} m³/s)",
    )

    flood_dates = test_dates_dt[actual_flood]
    flood_vals = targets_original[actual_flood]
    ax.scatter(
        flood_dates,
        flood_vals,
        c="red",
        s=15,
        alpha=0.6,
        zorder=5,
        label="Actual Flood Events",
    )

    detected_dates = test_dates_dt[actual_flood & predicted_flood]
    detected_vals = targets_original[actual_flood & predicted_flood]
    ax.scatter(
        detected_dates,
        detected_vals,
        c="green",
        s=30,
        alpha=0.8,
        zorder=6,
        marker="^",
        label="Correctly Detected",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title(
        f"Flood Event Detection — Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}"
    )
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "flood_detection.png"), dpi=150)
    plt.close(fig)
    print("Saved: flood_detection.png")

    print(f"\nAll results saved to {results_dir}")
    return metrics


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "synthetic_hydrology.csv"
    )
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    evaluate_model(csv_path=data_path, results_dir=results_dir)
