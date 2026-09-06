"""The pipeline diagram, drawn from the recorded run rather than written by hand.

The previous version was hand-written and had gone stale: it named a script that
no longer exists, quoted a best epoch and a set of scores from an earlier run, and
counted figures that had changed. Generating it removes the chance of that
happening again.

    python scripts/figures/generate_diagram.py

Output: docs/diagrams/pipeline.svg
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results" / "benchmark.json"
HPARAMS = ROOT / "results" / "hyperparameters.json"
OUT = ROOT / "docs" / "diagrams" / "pipeline.svg"

INK = "#111827"
MUTED = "#4B5563"
FAINT = "#9CA3AF"
HAIR = "#E5E7EB"
BLUE, BLUE_BG = "#2563EB", "#EFF6FF"
GREEN, GREEN_BG = "#059669", "#ECFDF5"
AMBER, AMBER_BG = "#D97706", "#FFFBEB"
GREY_BG = "#F9FAFB"
FONT = "Segoe UI, -apple-system, Helvetica, Arial, sans-serif"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, body, cls="sub", anchor="start"):
    return (f'  <text class="{cls}" x="{x}" y="{y}" '
            f'text-anchor="{anchor}">{esc(body)}</text>\n')


def box(x, y, w, h, fill, edge, title, lines):
    out = (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
           f'stroke="{edge}" stroke-width="1.4"/>\n'
           f'  <text class="lbl" x="{x + 14}" y="{y + 24}">{esc(title)}</text>\n')
    for i, line in enumerate(lines):
        out += text(x + 14, y + 45 + i * 16, line, "sub")
    return out


def arrow(x1, y1, x2, y2, colour=FAINT):
    head = 8.0
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    ex, ey = x2 - ux * head, y2 - uy * head
    px, py = -uy, ux
    return (f'  <line x1="{x1}" y1="{y1}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{colour}" '
            f'stroke-width="1.6"/>\n'
            f'  <polygon points="{x2},{y2} {ex + px * 4.4:.1f},{ey + py * 4.4:.1f} '
            f'{ex - px * 4.4:.1f},{ey - py * 4.4:.1f}" fill="{colour}"/>\n')


def main() -> int:
    if not BENCH.is_file():
        raise SystemExit("results/benchmark.json is missing. Run: python src/experiment.py")
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    hparams = json.loads(HPARAMS.read_text(encoding="utf-8"))

    split = bench["split"]
    lstm = bench["test_metrics"]["lstm"]
    clim = bench["test_metrics"]["climatology"]
    rain = bench["variance_share"]["quickflow"] * 100
    seasonal = bench["seasonal_combined_share"] * 100

    W, H = 1080, 610
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" font-family="{FONT}">\n'
           f'  <rect width="{W}" height="{H}" fill="#FFFFFF"/>\n'
           f"  <defs><style>\n"
           f"    .h {{ fill:{INK}; font-size:19px; font-weight:600; }}\n"
           f"    .s {{ fill:{MUTED}; font-size:12.5px; }}\n"
           f"    .lbl {{ fill:{INK}; font-size:13px; font-weight:600; }}\n"
           f"    .sub {{ fill:{MUTED}; font-size:10.5px; }}\n"
           f"    .cap {{ fill:{FAINT}; font-size:11px; }}\n"
           f"  </style></defs>\n")

    svg += text(30, 36, "From a synthetic catchment to a next-day forecast", "h")
    svg += text(30, 58, "Every box is code in this repository. The scores come from "
                        "results/benchmark.json.", "s")

    svg += box(30, 84, 235, 104, GREY_BG, HAIR, "generate_data.py", [
        f"{split['train'] + split['val'] + split['test']:,} daily steps, seed 42",
        "rain, temperature, soil bucket",
        f"seasonal terms {seasonal:.0f}% of variance",
        f"rain-driven term {rain:.1f}%",
    ])
    svg += arrow(269, 136, 316, 136)
    svg += box(320, 84, 235, 104, BLUE_BG, BLUE, "dataset.py", [
        f"{split['window']} day windows, 3 features",
        "target is the day after the window",
        f"{split['strategy']} 70 / 15 / 15 split",
        "scalers fitted on train only",
    ])
    svg += arrow(559, 136, 606, 136)
    svg += box(610, 84, 235, 104, GREY_BG, HAIR, "model.py and train.py", [
        f"2-layer LSTM, {hparams['hidden_size']} hidden",
        f"{bench['model']['parameters']:,} parameters",
        f"early stopping, best epoch {hparams['best_epoch']}",
        f"seed {hparams.get('seed')}",
    ])

    svg += arrow(727, 192, 727, 236)
    svg += box(610, 240, 235, 104, GREEN_BG, GREEN, "experiment.py", [
        f"scores {split['scored_rows']:,} held-out days",
        "model and every baseline",
        "on exactly the same rows",
        "writes results/benchmark.json",
    ])
    svg += arrow(606, 292, 559, 292)
    svg += box(320, 240, 235, 104, AMBER_BG, AMBER, "baselines.py", [
        "train mean, the floor",
        "day-of-year mean, calendar only",
        "linear model on the same window",
        "persistence, for context",
    ])
    svg += arrow(316, 292, 269, 292)
    svg += box(30, 240, 235, 104, GREY_BG, HAIR, "scripts/figures", [
        "five figures, all from the",
        "recorded run",
        "check_repository.py verifies",
        "the README still matches",
    ])

    svg += f'  <line x1="30" y1="382" x2="{W - 30}" y2="382" stroke="{HAIR}"/>\n'
    svg += text(30, 410, "What the run found", "lbl")
    svg += text(30, 434,
                f"The LSTM reaches R2 {lstm['r2']:.3f} on the held-out split. A lookup "
                f"table of the mean discharge for each calendar day reaches "
                f"{clim['r2']:.3f}", "cap")
    svg += text(30, 452,
                f"on the same rows, using no weather input at all. Independent daily "
                f"noise in the generator caps any forecaster at "
                f"{bench['noise_ceiling_r2']:.3f}.", "cap")
    svg += text(30, 478,
                f"The gap between the model and the calendar, {lstm['r2'] - clim['r2']:.3f}, "
                f"is the whole value the network adds. Most of the score is the "
                f"seasonal cycle,", "cap")
    svg += text(30, 496,
                "which the table learns too.", "cap")

    svg += text(30, 532, "Synthetic catchment. Not a real river, not a flood warning "
                         "system, and not validated for any operational use.", "s")
    svg += text(30, 560, "Generated by scripts/figures/generate_diagram.py.", "cap")
    svg += "</svg>\n"

    for chunk in svg.split("<text")[1:]:
        body = chunk.split(">", 1)[1].split("</text>")[0]
        assert "\n" not in body, "newline inside a text element"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8", newline="\n")
    print(f"  wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
