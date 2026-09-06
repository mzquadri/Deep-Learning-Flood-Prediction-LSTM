"""Verify the versioned artifacts this experiment advertises.

Cheap by design: it confirms the tracked source and result files exist and that
the benchmark JSON parses and still agrees with the README's headline claim. It
does not regenerate data or retrain, so it stays usable as a quick check.
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


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))

    bench = json.loads((ROOT / "results" / "benchmark.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # The one claim worth guarding here: the README must not quote a score the
    # recorded run does not contain, and must not drop the baseline it is
    # compared against.
    problems = []
    for name in ("lstm", "climatology"):
        value = f"{bench['test_metrics'][name]['r2']:.3f}"
        if value not in readme:
            problems.append(f"README does not state the {name} R2 of {value}")
    if problems:
        raise SystemExit("README disagrees with results/benchmark.json:\n"
                         + "\n".join(problems))

    print(f"Repository check passed: {len(REQUIRED)} artifacts present, "
          f"README matches results/benchmark.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
