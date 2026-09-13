"""Shared light-ground styling for this repository's figures."""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt

#: tEXt key holding digests of the inputs a figure was drawn from.
DRAWN_FROM_KEY = "DrawnFrom"

PAPER = "#FFFFFF"
INK = "#111827"
MUTED = "#4B5563"
FAINT = "#9CA3AF"
HAIR = "#E5E7EB"

BLUE, BLUE_SOFT = "#2563EB", "#BFDBFE"
GREEN, GREEN_SOFT = "#059669", "#A7F3D0"
AMBER, AMBER_SOFT = "#D97706", "#FDE68A"
RED, RED_SOFT = "#DC2626", "#FECACA"
SLATE, SLATE_SOFT = "#64748B", "#CBD5E1"

CYCLE = [BLUE, GREEN, AMBER, SLATE, RED]

#: Seven distinct hues, for the seven series. Kept light and non-neon.
BLUE_BG, GREEN_BG, AMBER_BG = "#EFF6FF", "#ECFDF5", "#FFFBEB"
TEAL, TEAL_SOFT = "#0891B2", "#A5F3FC"
FONTS = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.dpi": 200,
        "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": FONTS,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": HAIR,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
    })


def clean(ax, *, left: bool = True, bottom: bool = True, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=0)
    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, color=HAIR, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def fits(fig, artist, *, margin: float = 0.02) -> bool:
    """True if a drawn text artist stays inside the canvas."""
    fig.canvas.draw()
    box = artist.get_window_extent(fig.canvas.get_renderer())
    return box.x1 <= fig.bbox.x1 * (1.0 - margin) and box.x0 >= 0.0


def title_block(fig, title: str, subtitle: str = "", *, y: float = 0.96,
                size: float = 20, x: float = 0.065) -> None:
    fig.text(x, y, title, fontsize=size, color=INK, fontweight="600", va="top")
    if subtitle:
        gap = (size * 1.45) / (fig.get_figheight() * 72.0)
        fig.text(x, y - gap, subtitle, fontsize=11.2, color=MUTED, va="top",
                 linespacing=1.55)


def footnote(fig, lines, *, y: float = 0.085, x: float = 0.065, size: float = 9.4) -> None:
    step = (size * 1.75) / (fig.get_figheight() * 72.0)
    for i, line in enumerate(lines):
        artist = fig.text(x, y - i * step, line, fontsize=size, color=FAINT, va="top",
                          linespacing=1.5)
        if not fits(fig, artist):
            print(f"    caption overflows the canvas: {line[:60]}...")


def save(fig, out_dir, name: str, *, drawn_from: dict | None = None) -> None:
    """Write the figure, recording what it was drawn from.

    Every figure here recomputes the held-out predictions by loading
    results/best_model.pt and running a forward pass, so a figure depends on the
    checkpoint as well as on results/benchmark.json. `drawn_from` carries a
    digest of each, in the PNG's tEXt block, and scripts/check_repository.py
    recomputes them.

    Comparing the images byte for byte would not work, and that is measured
    rather than assumed: requirements.txt leaves torch as a lower bound because
    the build differs by accelerator, and rendering this set under 2.14.0+cpu
    reproduces four of the five files exactly while 04_error_analysis.png
    differs, because it plots residuals at a resolution where a last-bit
    difference in the forward pass moves a point. Every number in
    results/benchmark.json is identical between the two builds.

    matplotlib merges this with its own defaults, so the Software entry naming
    the version that rendered the file is still written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = None if drawn_from is None else {
        DRAWN_FROM_KEY: json.dumps(drawn_from, sort_keys=True)}
    fig.savefig(out_dir / f"{name}.png", metadata=metadata)
    plt.close(fig)
    print(f"  wrote {name}.png")
