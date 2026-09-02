"""
PyTorch Dataset and DataLoader utilities for flood prediction.

Handles:
- Loading and preprocessing the hydrological CSV data
- Sliding window sequence creation for LSTM input
- Train/validation/test splitting
- Feature normalization (StandardScaler)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import os
import joblib


FEATURE_COLS = ["precipitation_mm", "temperature_c", "soil_moisture_pct"]
TARGET_COL = "discharge_m3s"


class HydrologySequenceDataset(Dataset):
    """
    Sliding-window dataset for time series prediction.

    Each sample consists of `seq_len` consecutive days of features
    and the target value at the next time step (one-step-ahead prediction).
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 30):
        """
        Parameters
        ----------
        features : np.ndarray, shape (n_samples, n_features)
            Normalized feature array.
        targets : np.ndarray, shape (n_samples,)
            Normalized target array.
        seq_len : int
            Number of past time steps used as input.
        """
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.from_numpy(x), torch.tensor(y)


def load_and_split(
    csv_path: str,
    seq_len: int = 30,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    batch_size: int = 64,
    scaler_dir: str | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load CSV, normalize, create sequences, and return DataLoaders.

    Parameters
    ----------
    csv_path : str
        Path to the hydrological CSV file.
    seq_len : int
        Sequence length for LSTM input.
    train_frac : float
        Fraction of data for training.
    val_frac : float
        Fraction of data for validation.
    batch_size : int
        Batch size for DataLoaders.
    scaler_dir : str, optional
        Directory to save fitted scalers.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    info : dict
        Contains scalers, split indices, and metadata.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features = df[FEATURE_COLS].values
    targets = df[TARGET_COL].values.reshape(-1, 1)

    # Fit scalers on training portion only
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaler.fit(features[:n_train])
    target_scaler.fit(targets[:n_train])

    features_scaled = feature_scaler.transform(features)
    targets_scaled = target_scaler.transform(targets).flatten()

    # Save scalers
    if scaler_dir is not None:
        os.makedirs(scaler_dir, exist_ok=True)
        joblib.dump(feature_scaler, os.path.join(scaler_dir, "feature_scaler.pkl"))
        joblib.dump(target_scaler, os.path.join(scaler_dir, "target_scaler.pkl"))

    # Split
    train_feat = features_scaled[:n_train]
    train_targ = targets_scaled[:n_train]

    val_feat = features_scaled[n_train : n_train + n_val]
    val_targ = targets_scaled[n_train : n_train + n_val]

    test_feat = features_scaled[n_train + n_val :]
    test_targ = targets_scaled[n_train + n_val :]

    # Create datasets
    train_ds = HydrologySequenceDataset(train_feat, train_targ, seq_len)
    val_ds = HydrologySequenceDataset(val_feat, val_targ, seq_len)
    test_ds = HydrologySequenceDataset(test_feat, test_targ, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Date arrays for plotting
    test_dates = df["date"].values[n_train + n_val + seq_len :]

    info = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n - n_train - n_val,
        "seq_len": seq_len,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "test_dates": test_dates,
    }

    print(
        f"Dataset splits — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader, info
