"""Common figure style (matches the report: 3.45 in = one IEEE column, 7-8 pt sans-serif text)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # validated categorical slots 1-4
NAVY, INK, INK2, MUTED = "#1c5cab", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, GRAY = "#e1e0d9", "#c3c2b7", "#8f8d86"
WIDTH = 3.45

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.8, "axes.edgecolor": BASE, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def tidy(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{name}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png
