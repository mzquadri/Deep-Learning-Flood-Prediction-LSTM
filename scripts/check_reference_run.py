"""Check that a fresh run still produces the numbers this repository publishes.

The gap this closes: `src/experiment.py` writes `results/benchmark.json`
unconditionally. Run it after a change and it overwrites the recorded result and
exits 0, so a moved number leaves no trace in CI. The workflow step that runs it
claimed a change would fail there. It would not have. This script is what makes
that claim true.

It compares the working copy of the benchmark against the committed copy, field
by field, and fails on the first disagreement. Nothing is restated here, so
there is no hand-maintained list of expected numbers to drift out of date: the
committed file is the expectation.

    python src/generate_data.py
    python src/experiment.py
    python scripts/check_reference_run.py

`environment` is skipped. It records when the run happened and which torch
built it, which differ on every machine and say nothing about the result.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = "results/benchmark.json"

# Ignored because they are properties of the run, not of the result.
SKIP_TOP_LEVEL = {"environment"}

# The README quotes discharge to three decimals (2.416 m3/s) and the shares to
# one (77.2%). Half a unit in the last published digit is therefore the point at
# which a difference becomes visible to a reader, and that is the tolerance.
# Measured before it was chosen: regenerating the data and re-scoring the
# committed checkpoint reproduces every number in this file exactly, so the
# allowance exists only to absorb a different torch build, not real movement.
TOLERANCE = 5e-4


def committed_copy() -> dict:
    """The benchmark as recorded at HEAD, read without touching the index."""
    try:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{BENCHMARK}"],
            cwd=ROOT, capture_output=True, check=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            f"Cannot read {BENCHMARK} from HEAD, so there is nothing to compare "
            f"against: {exc}"
        ) from exc
    return json.loads(blob)


def compare(expected: object, actual: object, path: str, out: list[str]) -> None:
    """Walk both structures together, recording every place they disagree."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            out.append(f"{path}: expected an object, got {type(actual).__name__}")
            return
        for key in expected:
            if key not in actual:
                out.append(f"{path}/{key}: missing from the new run")
            else:
                compare(expected[key], actual[key], f"{path}/{key}", out)
        for key in actual:
            if key not in expected:
                out.append(f"{path}/{key}: present in the new run, absent from HEAD")
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            out.append(f"{path}: expected {expected!r}, got {actual!r}")
            return
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            compare(e, a, f"{path}[{i}]", out)
        return

    # bool first: it is a subclass of int, and True == 1 would pass numerically.
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            out.append(f"{path}: expected {expected!r}, got {actual!r}")
        return

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        # Counts are exact. A split of 5110 rows becoming 5109 is a real change
        # in the experiment, never a floating point artefact.
        if isinstance(expected, int) and isinstance(actual, int):
            if expected != actual:
                out.append(f"{path}: expected {expected}, got {actual}")
        elif abs(expected - actual) > TOLERANCE:
            out.append(
                f"{path}: expected {expected} +/- {TOLERANCE}, got {actual} "
                f"(moved by {actual - expected:+.6f})"
            )
        return

    if expected != actual:
        out.append(f"{path}: expected {expected!r}, got {actual!r}")


def main() -> int:
    current_path = ROOT / BENCHMARK
    if not current_path.is_file():
        print(f"No {BENCHMARK}. Run: python src/experiment.py", file=sys.stderr)
        return 1

    expected = committed_copy()
    actual = json.loads(current_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    for key in expected:
        if key in SKIP_TOP_LEVEL:
            continue
        if key not in actual:
            failures.append(f"/{key}: missing from the new run")
        else:
            compare(expected[key], actual[key], f"/{key}", failures)
    for key in actual:
        if key not in expected and key not in SKIP_TOP_LEVEL:
            failures.append(f"/{key}: present in the new run, absent from HEAD")

    if failures:
        print(
            "The run no longer reproduces the committed result. Either the change "
            "was not intended, or the README and the benchmark both need updating "
            "in the same commit:",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    metrics = actual["test_metrics"]
    print(
        "Fresh run matches the committed benchmark: "
        f"LSTM R2 {metrics['lstm']['r2']:.4f}, "
        f"day-of-year mean {metrics['climatology']['r2']:.4f}, "
        f"ceiling {actual['noise_ceiling_r2']:.4f}, "
        f"on {actual['split']['scored_rows']} scored rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
