"""Verify the versioned artifacts this experiment advertises.

Cheap by design: it confirms the tracked source and result files exist and that
every number the README publishes is still the number the recorded run produced.
It does not regenerate data or retrain, so it stays usable as a quick check.
For the stronger statement, that a fresh run still lands on those numbers, see
scripts/check_reference_run.py.

Each claim below is derived from results/benchmark.json and then looked for in
the README, formatted to the precision the README prints. No number is typed in
twice, so this file cannot disagree with the benchmark on its own.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "README.md",
    "requirements.txt",
    "src/generate_data.py",
    "src/dataset.py",
    "src/model.py",
    "src/train.py",
    "src/baselines.py",
    "src/experiment.py",
    "results/best_model.pt",
    "results/benchmark.json",
    "results/hyperparameters.json",
    "results/training_history.json",
    "docs/figures/01_signal_composition.png",
    "docs/figures/02_model_against_baselines.png",
    "docs/figures/03_held_out_forecast.png",
    "docs/figures/04_error_analysis.png",
    "docs/figures/05_training.png",
    "docs/diagrams/pipeline.svg",
)

# The names the README gives the baselines, so a failure message points at the
# row a reader would look for rather than at a dictionary key.
LABELS = {
    "lstm": "LSTM",
    "climatology": "day-of-year mean",
    "persistence": "persistence",
    "linear_regression": "linear regression",
    "train_mean": "train mean",
}


def claims(bench: dict) -> list[tuple[str, str]]:
    """Every published number, as (what it is, how the README prints it)."""
    out: list[tuple[str, str]] = []

    # The results table: five methods, three metrics each, three decimals.
    for key, label in LABELS.items():
        row = bench["test_metrics"][key]
        for metric in ("rmse", "mae", "r2"):
            out.append((f"the {label} {metric.upper()}", f"{row[metric]:.3f}"))

    lstm_r2 = bench["test_metrics"]["lstm"]["r2"]
    clim_r2 = bench["test_metrics"]["climatology"]["r2"]

    out.append(("the noise ceiling on R2", f"{bench['noise_ceiling_r2']:.3f}"))
    # The gap between the model and a lookup table of calendar-day means is the
    # README's central claim about how much the network is actually worth, and
    # it is a subtraction, so it can go stale while both of its inputs stay right.
    out.append(("the margin over the calendar", f"{lstm_r2 - clim_r2:.3f}"))

    out.append(("the combined seasonal share of variance",
                f"{bench['seasonal_combined_share'] * 100:.0f}%"))
    out.append(("the rain-driven share of variance",
                f"{bench['variance_share']['quickflow'] * 100:.1f}%"))
    out.append(("the share of days soil moisture sits at its floor",
                f"{bench['data']['soil_moisture_at_floor_fraction'] * 100:.0f}%"))

    # The event detection table, and the threshold it is computed at.
    detection = bench["event_detection"]
    for key in ("lstm", "persistence", "climatology", "linear_regression"):
        row = detection[key]
        for metric in ("precision", "recall", "f1"):
            out.append((f"the {LABELS[key]} {metric}", f"{row[metric]:.3f}"))
    out.append(("the event threshold", f"{detection['lstm']['threshold_m3s']:.2f}"))

    out.append(("the number of scored rows", str(bench["split"]["scored_rows"])))
    out.append(("the parameter count", f"{bench['model']['parameters']:,}"))
    return out


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))

    bench = json.loads((ROOT / "results" / "benchmark.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    checked = claims(bench)
    problems = [f"README does not state {what} of {value}"
                for what, value in checked if value not in readme]
    if problems:
        raise SystemExit("README disagrees with results/benchmark.json:\n"
                         + "\n".join(f"  - {p}" for p in problems))

    print(f"Repository check passed: {len(REQUIRED)} artifacts present, "
          f"{len(checked)} published numbers match results/benchmark.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
