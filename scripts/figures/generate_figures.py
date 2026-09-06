"""Figures for the benchmark, drawn from the generated data and the recorded run.

Four figures, each answering one question:

    01  what makes up the synthetic signal, and how much of it is rainfall
    02  does the LSTM beat the baselines, and how close is the ceiling
    03  what a held-out forecast looks like on an objectively chosen window
    04  where the errors are, and what they cost at the event threshold
    05  whether the training run itself was well behaved

    python scripts/figures/generate_figures.py

Output: docs/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import portfolio_style as ps  # noqa: E402
import torch  # noqa: E402

from baselines import climatology_baseline, linear_baseline, make_windows  # noqa: E402
from dataset import FEATURE_COLS, TARGET_COL, load_and_split  # noqa: E402
from generate_data import generate_hydrological_data  # noqa: E402
from model import FloodLSTM  # noqa: E402

OUT = ROOT / "docs" / "figures"
BENCH = ROOT / "results" / "benchmark.json"
CSV = ROOT / "data" / "synthetic_hydrology.csv"

NICE = {
    "lstm": "LSTM",
    "climatology": "Day-of-year mean",
    "persistence": "Persistence",
    "linear_regression": "Linear regression",
    "train_mean": "Train mean",
}


def load_bench() -> dict:
    if not BENCH.exists():
        raise SystemExit("results/benchmark.json is missing. Run: python src/experiment.py")
    return json.loads(BENCH.read_text(encoding="utf-8"))


def rebuild_predictions(bench):
    """Recompute the held-out predictions the figures draw."""
    seq_len = bench["split"]["window"]
    df, components = generate_hydrological_data(n_years=20, seed=42, return_components=True)
    _, _, test_loader, info = load_and_split(str(CSV), seq_len=seq_len, batch_size=64)

    model = FloodLSTM(input_size=len(FEATURE_COLS), hidden_size=128, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load(ROOT / "results" / "best_model.pt", map_location="cpu"))
    model.eval()
    preds, actual = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x).numpy())
            actual.append(y.numpy())
    scaler = info["target_scaler"]
    lstm = scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
    truth = scaler.inverse_transform(np.concatenate(actual).reshape(-1, 1)).flatten()

    y = df[TARGET_COL].values
    doy = pd.to_datetime(df["date"]).dt.dayofyear.values
    n_train, n_val = info["n_train"], info["n_val"]
    start = n_train + n_val
    features = df[FEATURE_COLS].values
    fs = info["feature_scaler"]
    x_tr, y_tr = make_windows(fs.transform(features[:n_train]), y[:n_train], seq_len)
    x_te, _ = make_windows(fs.transform(features[start:]), y[start:], seq_len)

    out = {
        "lstm": lstm,
        "climatology": climatology_baseline(doy[:n_train], y[:n_train], doy[start + seq_len:]),
        "linear_regression": linear_baseline(x_tr, y_tr, x_te),
        "persistence": y[start:][seq_len - 1:-1],
    }
    dates = pd.to_datetime(df["date"]).values[start + seq_len:]
    return df, components, truth, out, dates


def fig_generator(df, components, bench):
    """What creates the signal, and how little of it is rainfall."""
    fig = plt.figure(figsize=(13.0, 7.4))
    axL = fig.add_axes([0.070, 0.240, 0.475, 0.500])
    axR = fig.add_axes([0.645, 0.240, 0.300, 0.500])

    days = slice(365, 365 + 730)
    t = pd.to_datetime(df["date"]).values[days]
    stack = [("baseflow", ps.SLATE_SOFT), ("snowmelt", ps.BLUE_SOFT),
             ("quickflow", ps.GREEN), ("noise", ps.AMBER_SOFT)]
    bottom = np.zeros(len(t))
    for name, colour in stack:
        vals = components[name][days]
        axL.fill_between(t, bottom, bottom + vals, color=colour, lw=0, zorder=3,
                         label=name.replace("quickflow", "quickflow (rain)"))
        bottom = bottom + vals
    axL.plot(t, df[TARGET_COL].values[days], color=ps.INK, lw=1.0, zorder=5,
             label="discharge")
    ps.clean(axL)
    axL.set_ylabel("discharge (m3/s)", fontsize=10.4)
    axL.set_xlabel("two representative years", fontsize=10.4)
    axL.legend(fontsize=8.8, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.20))

    share = bench["variance_share"]
    names = sorted(share, key=lambda k: -share[k])
    colours = {"snowmelt": ps.BLUE, "baseflow": ps.SLATE, "quickflow": ps.GREEN,
               "noise": ps.AMBER}
    ypos = np.arange(len(names))[::-1]
    for i, name in enumerate(names):
        axR.barh(ypos[i], share[name] * 100, color=colours[name], height=0.6,
                 alpha=0.88, zorder=3)
        axR.text(-0.02, ypos[i], name.replace("quickflow", "quickflow (rain)"),
                 ha="right", va="center", fontsize=10.0, color=ps.INK,
                 transform=axR.get_yaxis_transform())
        axR.text(share[name] * 100 + 0.9, ypos[i], f"{share[name] * 100:.1f}%",
                 va="center", fontsize=9.6, color=colours[name], fontweight="600")
    axR.set_yticks([])
    axR.set_xlim(0, max(share.values()) * 100 * 1.28)
    ps.clean(axR, left=False, grid_axis="x")
    axR.set_xlabel("share of target variance", fontsize=10.4)
    axR.text(0, 1.06, "what the target is made of", transform=axR.transAxes,
             fontsize=10.6, color=ps.INK, fontweight="600", va="bottom")

    rain = share["quickflow"] * 100
    seasonal = bench["seasonal_combined_share"] * 100
    ps.title_block(
        fig, "The benchmark is mostly a seasonal cycle",
        "Discharge is built from four terms. Only one of them responds to "
        "rainfall, which is what a model reading a\n30-day precipitation window is "
        "supposed to exploit.", y=0.955, size=20)
    ps.footnote(fig, [
        f"The two seasonal terms together account for {seasonal:.0f}% of the variance "
        f"and the "
        f"rain-driven term for {rain:.1f}%. A forecaster can therefore score well "
        f"here by learning the calendar and little else.",
        "That is why the day-of-year baseline in the next figure is the one worth "
        "beating, and why a good score on this data is not evidence of "
        "rainfall-runoff skill.",
        "Source: the generator's own components, results/benchmark.json."], y=0.100)
    ps.save(fig, OUT, "01_signal_composition")


def fig_baselines(bench):
    """Does the model earn its place, and how far is the ceiling."""
    scored = bench["test_metrics"]
    order = [k for k in bench["ranking_by_rmse"] if k != "train_mean"]
    ceiling = bench["noise_ceiling_r2"]

    fig = plt.figure(figsize=(13.0, 7.2))
    axL = fig.add_axes([0.235, 0.250, 0.310, 0.490])
    axR = fig.add_axes([0.660, 0.250, 0.290, 0.490])

    colours = {"lstm": ps.BLUE, "climatology": ps.AMBER, "persistence": ps.SLATE,
               "linear_regression": ps.SLATE_SOFT}
    ypos = np.arange(len(order))[::-1]
    for i, name in enumerate(order):
        axL.barh(ypos[i], scored[name]["rmse"], color=colours.get(name, ps.SLATE),
                 height=0.58, alpha=0.9, zorder=3)
        axL.text(-0.12, ypos[i], NICE.get(name, name), ha="right", va="center",
                 fontsize=10.4, color=ps.INK, transform=axL.get_yaxis_transform())
        axL.text(scored[name]["rmse"] + 0.09, ypos[i], f"{scored[name]['rmse']:.3f}",
                 va="center", fontsize=9.8, color=colours.get(name, ps.SLATE),
                 fontweight="600")
    axL.set_yticks([])
    axL.set_xlim(0, max(scored[n]["rmse"] for n in order) * 1.22)
    ps.clean(axL, left=False, grid_axis="x")
    axL.set_xlabel("RMSE on the held-out split (m3/s), lower is better", fontsize=10.4)

    for i, name in enumerate(order):
        axR.barh(ypos[i], scored[name]["r2"], color=colours.get(name, ps.SLATE),
                 height=0.58, alpha=0.9, zorder=3)
        axR.text(scored[name]["r2"] + 0.012, ypos[i], f"{scored[name]['r2']:.3f}",
                 va="center", fontsize=9.8, color=colours.get(name, ps.SLATE),
                 fontweight="600")
    axR.axvline(ceiling, color=ps.RED, lw=1.4, ls="--", zorder=5)
    axR.text(ceiling, len(order) - 0.32, f"noise ceiling {ceiling:.3f}", fontsize=9.4,
             color=ps.RED, ha="right", va="bottom", rotation=90)
    axR.set_yticks([])
    axR.set_xlim(0, 1.14)
    ps.clean(axR, left=False, grid_axis="x")
    axR.set_xlabel("R2, higher is better", fontsize=10.4)

    lstm, clim = scored["lstm"], scored["climatology"]
    ps.title_block(
        fig, "The model beats the calendar, but not by much",
        "Every method is scored on the same held-out rows. The dashed line is the "
        "best any forecaster could do:\nthe generator adds independent noise each "
        "day that nothing can predict.", y=0.955, size=20)
    ps.footnote(fig, [
        f"The LSTM reaches R2 {lstm['r2']:.3f} against {clim['r2']:.3f} for a lookup "
        f"table of the mean discharge on each calendar day, which uses no weather "
        f"input at all. That gap is the whole value the model adds.",
        "Persistence is shown for context but is not a fair comparison: it uses "
        "yesterday's discharge, and discharge is not one of the model's inputs.",
        "Source: results/benchmark.json."], y=0.100)
    ps.save(fig, OUT, "02_model_against_baselines")


def fig_forecast(truth, preds, dates, bench):
    """A held-out window chosen by rule, not by eye."""
    # The window containing the largest observed event, so the figure shows the
    # hardest part of the split rather than a flattering quiet stretch.
    peak = int(np.argmax(truth))
    half = 90
    lo = max(0, min(peak - half, len(truth) - 2 * half))
    hi = lo + 2 * half
    window = slice(lo, hi)

    fig = plt.figure(figsize=(13.0, 6.8))
    ax = fig.add_axes([0.070, 0.255, 0.885, 0.475])
    t = dates[window]

    threshold = bench["event_detection"]["lstm"]["threshold_m3s"]
    ax.axhline(threshold, color=ps.RED, lw=1.1, ls="--", zorder=2)
    ax.text(t[2], threshold + 0.5, f"event threshold, {threshold:.1f} m3/s",
            fontsize=9.2, color=ps.RED, va="bottom")

    ax.plot(t, truth[window], color=ps.INK, lw=1.5, zorder=5, label="observed")
    ax.plot(t, preds["lstm"][window], color=ps.BLUE, lw=1.7, zorder=4, label="LSTM")
    ax.plot(t, preds["climatology"][window], color=ps.AMBER, lw=1.3, ls="--", zorder=3,
            label="day-of-year mean")
    ps.clean(ax)
    ax.set_ylabel("discharge (m3/s)", fontsize=10.4)
    ax.legend(fontsize=9.4, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.16))

    ps.title_block(
        fig, "The model tracks the events the calendar cannot",
        "A 180 day stretch of the held-out split, centred on the largest observed "
        "event. Chosen by that rule rather\nthan by eye.", y=0.955, size=20)
    ps.footnote(fig, [
        "The day-of-year baseline reproduces the seasonal shape. Its day-to-day "
        "movement is sampling noise from the training years, not a response to "
        "weather, so it cannot follow an individual storm.",
        "The LSTM does follow some of them, and that is where its advantage over "
        "the calendar comes from.",
        "It still under-predicts the sharpest peaks, which is the error pattern the "
        "next figure quantifies.",
        "Source: results/benchmark.json and the seed 42 generator."], y=0.100)
    ps.save(fig, OUT, "03_held_out_forecast")


def fig_errors(truth, preds, bench):
    """Where the error sits, and what it costs at the event threshold."""
    lstm = preds["lstm"]
    err = lstm - truth
    fig = plt.figure(figsize=(13.0, 7.0))
    axL = fig.add_axes([0.070, 0.250, 0.375, 0.480])
    axR = fig.add_axes([0.570, 0.250, 0.375, 0.480])

    axL.scatter(truth, err, s=9, color=ps.BLUE, alpha=0.35, zorder=3, edgecolors="none")
    axL.axhline(0, color=ps.HAIR, lw=1.2, zorder=2)
    threshold = bench["event_detection"]["lstm"]["threshold_m3s"]
    axL.axvline(threshold, color=ps.RED, lw=1.1, ls="--", zorder=4)
    axL.text(threshold + 0.4, axL.get_ylim()[1] * 0.92, "event threshold", fontsize=9.0,
             color=ps.RED, va="top")
    ps.clean(axL)
    axL.set_xlabel("observed discharge (m3/s)", fontsize=10.4)
    axL.set_ylabel("prediction minus observed (m3/s)", fontsize=10.4)
    axL.text(0, 1.06, "error against magnitude", transform=axL.transAxes, fontsize=10.6,
             color=ps.INK, fontweight="600", va="bottom")

    high = truth >= threshold
    bias_high = float(err[high].mean())
    bias_low = float(err[~high].mean())

    detect = bench["event_detection"]
    names = ["lstm", "persistence", "climatology", "linear_regression"]
    names = [n for n in names if n in detect]
    x = np.arange(len(names))
    width = 0.27
    for k, (metric, colour) in enumerate((("precision", ps.BLUE), ("recall", ps.GREEN),
                                          ("f1", ps.SLATE))):
        vals = [detect[n][metric] for n in names]
        axR.bar(x + (k - 1) * width, vals, width=width, color=colour, alpha=0.88,
                zorder=3, label=metric.replace("f1", "F1"))
    axR.set_xticks(x)
    axR.set_xticklabels([NICE.get(n, n).replace(" ", "\n") for n in names], fontsize=9.2)
    axR.set_ylim(0, 1.12)
    ps.clean(axR)
    axR.legend(fontsize=9.0, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.16))
    axR.text(1.0, 1.06, f"events above {threshold:.1f} m3/s", transform=axR.transAxes,
             fontsize=9.6, color=ps.FAINT, ha="right", va="bottom")

    ps.title_block(
        fig, "The errors grow with the events",
        "Residuals against observed discharge, and what that means when the task is "
        "turned into detecting the top\nfive percent of days.", y=0.955, size=20)
    ps.footnote(fig, [
        f"Below the threshold the model is close to unbiased ({bias_low:+.2f} m3/s "
        f"on average). Above it the mean error is {bias_high:+.2f} m3/s, so the "
        f"largest flows are systematically under-predicted.",
        f"That shows up directly in detection: recall "
        f"{detect['lstm']['recall']:.2f} against precision "
        f"{detect['lstm']['precision']:.2f}. The model misses events more often than "
        f"it invents them.",
        "Source: results/benchmark.json."], y=0.100)
    ps.save(fig, OUT, "04_error_analysis")
    return bias_low, bias_high


def fig_training(bench):
    """Did the run behave, and where did early stopping land."""
    history_path = ROOT / "results" / "training_history.json"
    hparams_path = ROOT / "results" / "hyperparameters.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(13.0, 6.4))
    axL = fig.add_axes([0.075, 0.270, 0.375, 0.450])
    axR = fig.add_axes([0.575, 0.270, 0.375, 0.450])

    axL.plot(epochs, history["train_loss"], color=ps.BLUE, lw=1.8, label="train")
    axL.plot(epochs, history["val_loss"], color=ps.AMBER, lw=1.8, label="validation")
    best = hparams["best_epoch"]
    axL.axvline(best, color=ps.INK, lw=1.1, ls=":", zorder=5)
    axL.text(best + 0.6, max(history["train_loss"]) * 0.86,
             f"checkpoint kept\nfrom epoch {best}", fontsize=9.2, color=ps.INK)
    ps.clean(axL)
    axL.set_xlabel("epoch", fontsize=10.4)
    axL.set_ylabel("mean squared error (scaled units)", fontsize=10.4)
    axL.legend(fontsize=9.4, ncol=2)

    axR.plot(epochs, history["lr"], color=ps.SLATE, lw=1.8)
    ps.clean(axR)
    axR.set_xlabel("epoch", fontsize=10.4)
    axR.set_ylabel("learning rate", fontsize=10.4)
    axR.text(0, 1.06, "the scheduler stepped down twice", transform=axR.transAxes,
             fontsize=10.4, color=ps.INK, fontweight="600", va="bottom")

    ps.title_block(
        fig, "The run stopped where it stopped improving",
        "Training and validation loss in scaled units, and the learning rate the "
        "scheduler chose.", y=0.955, size=20)
    ps.footnote(fig, [
        f"Validation loss is noisy after epoch {best} and does not improve on it, "
        f"which is what early stopping is for. The checkpoint scored everywhere else "
        f"in this repository is the one from that epoch, not the last one.",
        f"Trained on {bench['split']['train']} days with a {bench['split']['window']} "
        f"day window, seed {hparams.get('seed')}. Source: "
        f"results/training_history.json.", ], y=0.100)
    ps.save(fig, OUT, "05_training")


def main() -> int:
    ps.apply()
    bench = load_bench()
    df, components, truth, preds, dates = rebuild_predictions(bench)
    print(f"\n  {len(truth)} held-out rows\n")
    fig_generator(df, components, bench)
    fig_baselines(bench)
    fig_forecast(truth, preds, dates, bench)
    lo, hi = fig_errors(truth, preds, bench)
    fig_training(bench)
    print(f"\n  mean error below threshold {lo:+.3f}, above {hi:+.3f}")
    print(f"  figures written to {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
