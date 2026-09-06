"""Tests for the parts of this benchmark that would be silent if they broke.

Two of these exist because of defects that were actually present. The bucket
model was mis-scaled and pinned soil moisture against its lower bound on 82% of
days, which made the feature inert. And a benchmark with no baseline invites a
score to be read as a success when a lookup table does better.

The rest guard the properties a time-series experiment has to hold: the window
never contains its own target, the split is chronological, and the scaler never
sees the test years.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import (  # noqa: E402
    climatology_baseline,
    make_windows,
    metrics,
    noise_ceiling,
    persistence_baseline,
)
from dataset import FEATURE_COLS, HydrologySequenceDataset, load_and_split  # noqa: E402
from generate_data import generate_hydrological_data  # noqa: E402

torch = pytest.importorskip("torch")
from model import FloodLSTM  # noqa: E402

# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    return generate_hydrological_data(n_years=5, seed=42)


def test_the_generator_is_deterministic():
    a = generate_hydrological_data(n_years=3, seed=42)
    b = generate_hydrological_data(n_years=3, seed=42)

    pd.testing.assert_frame_equal(a, b)


def test_a_different_seed_gives_different_data():
    a = generate_hydrological_data(n_years=3, seed=42)
    b = generate_hydrological_data(n_years=3, seed=7)

    assert not a["discharge_m3s"].equals(b["discharge_m3s"])


def test_soil_moisture_does_not_sit_against_its_bound(data):
    """The bucket was mis-scaled and pinned this to the floor on 82% of days.

    A feature clamped at a clip bound carries no information, and it silently
    removed the nonlinear rainfall-runoff response it was supposed to drive.
    """
    at_floor = (data["soil_moisture_pct"] <= 5.0001).mean()

    assert at_floor < 0.35, f"soil moisture sits at its floor on {at_floor:.0%} of days"
    assert data["soil_moisture_pct"].std() > 5.0


def test_the_components_reconstruct_the_target():
    df, parts = generate_hydrological_data(n_years=3, seed=42, return_components=True)
    total = sum(parts.values())

    # The generator applies a physical floor of 1.0 after summing.
    assert np.allclose(np.maximum(total, 1.0), df["discharge_m3s"].values, atol=0.011)


def test_discharge_never_goes_negative(data):
    assert (data["discharge_m3s"] > 0).all()


# --------------------------------------------------------------------------
# windowing and the split
# --------------------------------------------------------------------------

def test_the_window_never_contains_its_own_target():
    """Sample i reads rows i to i+29 and predicts row i+30, one step ahead."""
    features = np.arange(100 * 3, dtype=float).reshape(100, 3)
    targets = np.arange(100, dtype=float)
    ds = HydrologySequenceDataset(features, targets, seq_len=30)

    x, y = ds[0]
    assert x.shape == (30, 3)
    assert float(y) == 30.0
    assert np.allclose(x.numpy(), features[0:30])
    assert len(ds) == 70


def test_make_windows_matches_the_torch_dataset():
    """The baselines must be scored on exactly the rows the model is scored on."""
    features = np.random.RandomState(0).randn(80, 3)
    targets = np.random.RandomState(1).randn(80)
    ds = HydrologySequenceDataset(features, targets, seq_len=30)
    x, y = make_windows(features, targets, 30)

    assert len(x) == len(ds)
    assert np.allclose(y, [float(ds[i][1]) for i in range(len(ds))])
    assert np.allclose(x[0], features[0:30].ravel())


def test_the_split_is_chronological_and_the_scaler_sees_only_training(tmp_path):
    df = generate_hydrological_data(n_years=6, seed=42)
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    _, _, _, info = load_and_split(str(csv), seq_len=30, batch_size=16)

    n = len(df)
    assert info["n_train"] == int(0.7 * n)
    assert info["n_val"] == int(0.15 * n)

    # The fitted mean must match the training rows, not the whole record.
    train_mean = df[FEATURE_COLS].values[: info["n_train"]].mean(axis=0)
    assert np.allclose(info["feature_scaler"].mean_, train_mean)
    whole_mean = df[FEATURE_COLS].values.mean(axis=0)
    assert not np.allclose(info["feature_scaler"].mean_, whole_mean)


def test_windows_do_not_straddle_a_split_boundary(tmp_path):
    """Each split builds windows from its own slice, so none spans a boundary."""
    df = generate_hydrological_data(n_years=6, seed=42)
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    train, val, test, info = load_and_split(str(csv), seq_len=30, batch_size=16)

    seq = info["seq_len"]
    assert len(train.dataset) == info["n_train"] - seq
    assert len(val.dataset) == info["n_val"] - seq
    assert len(test.dataset) == info["n_test"] - seq


# --------------------------------------------------------------------------
# baselines and metrics
# --------------------------------------------------------------------------

def test_climatology_uses_only_training_years():
    doy_train = np.tile(np.arange(1, 366), 3)
    y_train = np.tile(np.arange(1, 366, dtype=float), 3)
    predicted = climatology_baseline(doy_train, y_train, np.array([5, 100, 365]))

    assert np.allclose(predicted, [5.0, 100.0, 365.0])


def test_climatology_falls_back_for_an_unseen_day():
    predicted = climatology_baseline(np.array([1, 2]), np.array([10.0, 20.0]),
                                     np.array([200]))

    assert predicted[0] == pytest.approx(15.0)


def test_persistence_returns_the_previous_day():
    targets = np.arange(40, dtype=float)
    predicted = persistence_baseline(targets, seq_len=30)

    # Target i is targets[30 + i]; persistence offers targets[29 + i].
    assert np.allclose(predicted, targets[29:-1])
    assert len(predicted) == len(targets) - 30


def test_a_perfect_prediction_scores_perfectly():
    y = np.array([1.0, 5.0, 3.0, 9.0])
    m = metrics(y, y)

    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["pbias_percent"] == pytest.approx(0.0)


def test_predicting_the_mean_scores_zero_r2():
    y = np.array([1.0, 5.0, 3.0, 9.0])
    m = metrics(y, np.full_like(y, y.mean()))

    assert m["r2"] == pytest.approx(0.0)


def test_noise_ceiling_is_below_one():
    assert noise_ceiling(2.0, 10.0) == pytest.approx(0.8)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

def test_the_model_returns_one_value_per_sequence():
    model = FloodLSTM(input_size=3, hidden_size=16, num_layers=2, dropout=0.1)
    out = model(torch.randn(8, 30, 3))

    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_the_loss_is_finite_on_a_forward_and_backward_pass():
    model = FloodLSTM(input_size=3, hidden_size=16, num_layers=1, dropout=0.0)
    loss = torch.nn.MSELoss()(model(torch.randn(4, 30, 3)), torch.randn(4))
    loss.backward()

    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
