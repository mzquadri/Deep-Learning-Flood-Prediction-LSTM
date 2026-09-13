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

import hashlib
import json
import struct
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


#: Written by scripts/figures/portfolio_style.py into every figure it saves.
DRAWN_FROM_KEY = "DrawnFrom"

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
NUL = bytes([0x00])


def png_text(path: Path) -> dict[str, str]:
    """The tEXt entries of a PNG, read without a third-party imaging library."""
    raw = path.read_bytes()
    if raw[:len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise SystemExit(f"{path.name} is not a PNG")
    entries, offset = {}, len(PNG_SIGNATURE)
    while offset + 8 <= len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        if kind == b"tEXt":
            key, _, value = raw[offset + 8:offset + 8 + length].partition(NUL)
            entries[key.decode("latin-1")] = value.decode("latin-1")
        elif kind == b"IEND":
            break
        offset += 12 + length
    return entries


def expected_provenance(bench: dict) -> dict:
    """The digests a figure drawn from the committed state would carry."""
    semantic = {k: v for k, v in bench.items() if k != "environment"}
    canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
    return {
        "benchmark": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "checkpoint": hashlib.sha256(
            (ROOT / "results" / "best_model.pt").read_bytes()).hexdigest()[:16],
    }


def invariant_problems(bench: dict) -> list[str]:
    """Relationships the recorded run has to satisfy whatever the numbers are.

    These hold by construction rather than by measurement, so a violation is a
    fault in the benchmark rather than an interesting result. Checking them is
    independent of the README: a run could agree with the text in every digit
    and still be internally inconsistent.
    """
    problems: list[str] = []

    ceiling = bench["noise_ceiling_r2"]
    noise = bench["variance_share"]["noise"]
    if abs(ceiling - (1.0 - noise)) > 5e-4:
        problems.append(f"the ceiling is {ceiling} but the noise share is {noise}, "
                        f"and the two must satisfy ceiling = 1 - noise share")

    # The ceiling is the whole reason the baselines are reported. A method above
    # it would be predicting the generator's own unpredictable noise.
    for name, scores in bench["test_metrics"].items():
        if scores["r2"] > ceiling + 1e-9:
            problems.append(f"{LABELS.get(name, name)} scores R2 {scores['r2']}, "
                            f"above the noise ceiling of {ceiling}")

    # Var(baseflow + snowmelt) exceeds the sum of their variances only if the two
    # are positively correlated, which is what makes the combined figure the one
    # worth quoting. Both are recorded, so the relation is checkable.
    shares = bench["variance_share"]
    apart = shares["baseflow"] + shares["snowmelt"]
    together = bench["seasonal_combined_share"]
    if together < apart:
        problems.append(f"the combined seasonal share {together} is below the sum "
                        f"of the individual shares {apart:.4f}")

    if not problems:
        print(f"Invariants hold: ceiling {ceiling} = 1 - noise share, no method "
              f"above it, seasonal terms combine constructively.")
    return problems


def figure_problems(bench: dict) -> list[str]:
    """Each committed figure against the run and the checkpoint it came from.

    The figures were checked for existence and nothing else. Every one of them
    reloads results/best_model.pt and recomputes the held-out predictions, so a
    figure can be stale with respect to the checkpoint as well as to the
    benchmark, and continuous integration regenerates them without comparing.

    Pixels are the wrong comparison and that was measured: rendering this set
    under a different torch build reproduces four of the five files byte for
    byte and changes 04_error_analysis.png, while every number in the benchmark
    stays identical. What is compared is the digest of the inputs.
    """
    problems: list[str] = []
    figures = sorted((ROOT / "docs" / "figures").glob("*.png"))
    if not figures:
        return ["no figures in docs/figures"]

    expected = expected_provenance(bench)
    renderers = set()
    for figure in figures:
        text = png_text(figure)
        renderers.add(text.get("Software", "unrecorded"))
        if DRAWN_FROM_KEY not in text:
            problems.append(f"{figure.name} records nothing about what it was "
                            f"drawn from; rerun scripts/figures/generate_figures.py")
            continue
        stamped = json.loads(text[DRAWN_FROM_KEY])
        for what in ("benchmark", "checkpoint"):
            if stamped.get(what) != expected[what]:
                problems.append(
                    f"{figure.name} was drawn from a different {what}; "
                    f"rerun scripts/figures/generate_figures.py")
    if len(renderers) > 1:
        problems.append(f"the figures were not rendered together: {sorted(renderers)}")
    if not problems:
        print(f"Figure check passed: {len(figures)} figures carry the committed "
              f"benchmark and checkpoint.")
    return problems


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
    # Quoted beside the combined figure to show the two do not add, which is the
    # point of reporting the combined one.
    apart = (bench["variance_share"]["baseflow"]
             + bench["variance_share"]["snowmelt"])
    out.append(("the sum of the two seasonal shares taken separately",
                f"{apart * 100:.1f}%"))
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
    problems += figure_problems(bench)
    problems += invariant_problems(bench)
    if problems:
        raise SystemExit("README or figures disagree with the recorded run:\n"
                         + "\n".join(f"  - {p}" for p in problems))

    print(f"Repository check passed: {len(REQUIRED)} artifacts present, "
          f"{len(checked)} published numbers match results/benchmark.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
