"""Baselines the LSTM has to beat, and the ceiling nothing can beat.

A regression score means little on its own. R2 of 0.90 sounds strong until a
lookup table scores higher on the same split, and on this benchmark one does.

Three baselines are computed, chosen because each isolates a different thing:

    train mean          the floor. Anything below it has learned nothing.
    day-of-year mean    the seasonal cycle alone, no weather input at all.
    linear regression   the same 30-day window the LSTM reads, fitted linearly.

Persistence is reported separately because it is not a fair comparison: it uses
yesterday's discharge, and discharge is not one of the model's inputs. It is
included because a reader will ask.

The noise ceiling comes from the generator itself. Its measurement noise is drawn
independently each day and cannot be predicted from anything, so it caps the R2
any forecaster can reach on this data.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R2 and the hydrological pair NSE and PBIAS.

    NSE and R2 are the same quantity here. Both are reported because the
    hydrology literature names it NSE and the machine learning literature names
    it R2, and a reader from either side looks for their own name.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    ss_res = float((err**2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    nse = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": nse,
        "nse": nse,
        "pbias_percent": float(err.sum() / actual.sum() * 100.0) if actual.sum() else float("nan"),
        "n": len(actual),
    }


def make_windows(features: np.ndarray, targets: np.ndarray, seq_len: int):
    """Flatten each sliding window, matching what the LSTM is shown.

    Sample i reads features[i : i + seq_len] and predicts targets[i + seq_len],
    so the target day is never inside its own input window.
    """
    x = np.stack([features[i:i + seq_len].ravel() for i in range(len(features) - seq_len)])
    y = targets[seq_len:len(features)]
    return x, y


def train_mean_baseline(train_targets: np.ndarray, n_test: int) -> np.ndarray:
    return np.full(n_test, float(np.mean(train_targets)))


def climatology_baseline(train_doy: np.ndarray, train_targets: np.ndarray,
                         test_doy: np.ndarray) -> np.ndarray:
    """Mean discharge for each calendar day, fitted on the training years only.

    No weather input at all. On a seasonally driven signal this is hard to beat,
    which is exactly why it belongs here.
    """
    overall = float(np.mean(train_targets))
    table = np.full(367, overall)
    for day in np.unique(train_doy):
        table[int(day)] = float(np.mean(train_targets[train_doy == day]))
    return table[test_doy.astype(int)]


def linear_baseline(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Ordinary least squares on the flattened window."""
    return LinearRegression().fit(x_train, y_train).predict(x_test)


def persistence_baseline(test_targets_with_history: np.ndarray, seq_len: int) -> np.ndarray:
    """Yesterday's discharge. Uses an input the model does not receive."""
    return test_targets_with_history[seq_len - 1:-1]


def noise_ceiling(noise_variance: float, target_variance: float) -> float:
    """The highest R2 attainable when the only unpredictable part is the noise."""
    return 1.0 - noise_variance / target_variance
