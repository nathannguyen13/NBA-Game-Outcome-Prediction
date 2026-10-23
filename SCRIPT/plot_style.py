"""
plot_style.py
=============
Shared matplotlib styling imported by the numbered scripts. It is a helper module,
not a step in the pipeline, so it has no number and does not need to be run on its own.

It fixes one color per team (the color follows the team in every figure) and gives
all figures the same quiet axes, light grid, and font sizes.
"""

import matplotlib
matplotlib.use("Agg")            # write image files without needing a display
import matplotlib.pyplot as plt

# One fixed color per team, in a fixed order (colorblind-checked categorical palette).
TEAM_COLORS = {
    "BOS": "#2a78d6",   # blue
    "TOR": "#eb6834",   # orange
    "LAL": "#1baf7a",   # aqua
    "SAS": "#eda100",   # yellow
    "OKC": "#e87ba4",   # magenta
}
TEAM_ORDER = list(TEAM_COLORS)

WIN_COLOR = "#2a78d6"     # outcome encoding: blue = win
LOSS_COLOR = "#eb6834"    # orange = loss
NEUTRAL = "#8a8984"       # reference lines, baselines
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def apply() -> None:
    """Apply the project-wide figure style."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c9c8c2",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
