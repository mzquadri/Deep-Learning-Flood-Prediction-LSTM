"""Run the whole benchmark: the LSTM, the baselines, and the ceiling.

    python src/experiment.py

Reads the generated CSV, evaluates the trained checkpoint on the held-out split,
computes every baseline on the same split, and writes results/benchmark.json.

The point of running the baselines in the same place as the model is that a score
in isolation says nothing. This benchmark is dominated by its seasonal terms, and
a day-of-year lookup table is a genuinely hard thing to beat on it. That is worth
knowing before reading an R2.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import (
    climatology_baseline,
    linear_baseline,
    make_windows,
    metrics,
    noise_ceiling,
    persistence_baseline,
    train_mean_baseline,
)
from dataset import FEATURE_COLS, TARGET_COL, load_and_split
from generate_data import generate_hydrological_data
from model import FloodLSTM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "synthetic_hydrology.csv")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "benchmark.json")

#: The flood threshold used for the detection metrics, as a percentile of the
#: observed test discharge. Not a hydrological definition, a convention for
#: turning a regression into a rare-event detection problem.
FLOOD_PERCENTILE = 95


def lstm_predictions(info, seq_len: int, hparams: dict):
    """Run the saved checkpoint over the test split, in original units."""
    _, _, test_loader, _ = load_and_split(CSV, seq_len=seq_len,
                                          batch_size=hparams.get("batch_size", 64))
    model = FloodLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=hparams["hidden_size"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    )
    state = torch.load(os.path.join(RESULTS, "best_model.pt"), map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()

    preds, actual = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x).numpy())
            actual.append(y.numpy())
    scaler = info["target_scaler"]
    p = scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
    a = scaler.inverse_transform(np.concatenate(actual).reshape(-1, 1)).flatten()
    return p, a, model.count_parameters()


def detection(actual: np.ndarray, predicted: np.ndarray, threshold: float) -> dict:
    """Treat the top percentile of observed discharge as the event to catch."""
    true_event = actual >= threshold
    pred_event = predicted >= threshold
    tp = int((true_event & pred_event).sum())
    fp = int((~true_event & pred_event).sum())
    fn = int((true_event & ~pred_event).sum())
    tn = int((~true_event & ~pred_event).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold_m3s": round(float(threshold), 3), "true_positives": tp,
            "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "events": int(true_event.sum())}


def main() -> int:
    with open(os.path.join(RESULTS, "hyperparameters.json"), encoding="utf-8") as f:
        hparams = json.load(f)
    seq_len = hparams["seq_len"]

    df, components = generate_hydrological_data(n_years=20, seed=42, return_components=True)
    _, _, _, info = load_and_split(CSV, seq_len=seq_len, batch_size=hparams["batch_size"])

    y = df[TARGET_COL].values
    features = df[FEATURE_COLS].values
    doy = pd.to_datetime(df["date"]).dt.dayofyear.values
    n = len(df)
    n_train, n_val = info["n_train"], info["n_val"]
    test_start = n_train + n_val

    print(f"  {n} days, split {n_train} / {n_val} / {n - test_start}, window {seq_len}\n")

    # Every method is scored on exactly the same rows: the test targets that
    # follow a complete window inside the test split.
    fitted = info["feature_scaler"]
    x_train, y_train = make_windows(fitted.transform(features[:n_train]), y[:n_train], seq_len)
    x_test, y_test = make_windows(fitted.transform(features[test_start:]), y[test_start:], seq_len)
    test_doy = doy[test_start + seq_len:]

    preds = {}
    preds["lstm"], actual, n_params = lstm_predictions(info, seq_len, hparams)
    assert np.allclose(actual, y_test, atol=0.02), "baseline and model rows disagree"

    preds["linear_regression"] = linear_baseline(x_train, y_train, x_test)
    preds["climatology"] = climatology_baseline(doy[:n_train], y[:n_train], test_doy)
    preds["train_mean"] = train_mean_baseline(y_train, len(y_test))
    preds["persistence"] = persistence_baseline(y[test_start:], seq_len)

    scored = {name: metrics(y_test, p) for name, p in preds.items()}

    ceiling = noise_ceiling(float(components["noise"].var()), float(y.var()))
    variance = {
        name: round(float(comp.var()) / float(y.var()), 4)
        for name, comp in components.items()
    }
    # Baseflow and snowmelt are both driven by the calendar and are positively
    # correlated, so their combined share is larger than the sum of the two
    # individual shares. Summing them understates how seasonal this target is.
    seasonal_combined = float(
        (components["baseflow"] + components["snowmelt"]).var() / y.var())

    order = sorted(scored, key=lambda k: scored[k]["rmse"])
    print("  held-out test, ranked by RMSE:")
    for name in order:
        m = scored[name]
        note = ("  (uses past discharge, which the model never sees)"
                if name == "persistence" else "")
        print(f"    {name:20} RMSE {m['rmse']:6.3f}  MAE {m['mae']:6.3f}  "
              f"R2 {m['r2']:7.4f}{note}")
    print(f"\n  noise ceiling on R2: {ceiling:.4f}")
    print(f"  seasonal terms combined: {seasonal_combined*100:.1f}% of variance")
    print("  share of target variance by generator component:")
    for name, share in sorted(variance.items(), key=lambda kv: -kv[1]):
        print(f"    {name:12} {share * 100:5.1f}%")

    threshold = float(np.percentile(y_test, FLOOD_PERCENTILE))
    detect = {name: detection(y_test, p, threshold)
              for name, p in preds.items() if name != "train_mean"}
    print(f"\n  event detection above the {FLOOD_PERCENTILE}th percentile "
          f"({threshold:.2f} m3/s):")
    for name in sorted(detect, key=lambda k: -detect[k]["f1"]):
        d = detect[name]
        print(f"    {name:20} precision {d['precision']:.3f}  recall {d['recall']:.3f}  "
              f"F1 {d['f1']:.3f}")

    payload = {
        "environment": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "torch": torch.__version__,
        },
        "data": {
            "days": int(n),
            "years": int(n / 365),
            "seed": 42,
            "features": FEATURE_COLS,
            "target": TARGET_COL,
            "soil_moisture_at_floor_fraction": round(
                float((df["soil_moisture_pct"] <= 5.0001).mean()), 4),
        },
        "split": {"strategy": "chronological", "train": int(n_train), "val": int(n_val),
                  "test": int(n - test_start), "window": seq_len,
                  "scored_rows": len(y_test)},
        "model": {"architecture": "2-layer LSTM, 128 hidden, dense head",
                  "parameters": int(n_params),
                  "best_epoch": hparams.get("best_epoch"),
                  "seed": hparams.get("seed")},
        "variance_share": variance,
        "seasonal_combined_share": round(seasonal_combined, 4),
        "noise_ceiling_r2": round(float(ceiling), 4),
        "test_metrics": {k: {m: round(v, 4) for m, v in scored[k].items()} for k in scored},
        "event_detection": detect,
        "ranking_by_rmse": order,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\n  wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
