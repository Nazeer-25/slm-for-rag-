"""Phase 4 — figures for the comparison (one metric per panel, shared setup colors).

Usage:  python eval/plot_results.py            # reads results/comparison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order (validated palette, light mode). Color follows the
# setup, never its rank, so every figure paints each setup the same way.
SETUP_COLORS = {
    "undefended": "#2a78d6", "delimiter": "#eb6834", "llm_judge": "#1baf7a",
    "hybrid": "#eda100", "l1_drop": "#e87ba4", "l2_only": "#008300",
}
SETUP_LABELS = {
    "undefended": "Undefended", "delimiter": "Delimiter/XML", "llm_judge": "LLM-as-Judge",
    "hybrid": "Hybrid L1+L2 (ours)", "l1_drop": "L1 only (drop)", "l2_only": "L2 only (rules)",
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

PANELS = [
    ("ASR_%", "Attack success rate (%)", "lower is better"),
    ("utility_benign_%", "Benign task utility (%)", "higher is better"),
    ("fact_retention_rougeL_recall", "Fact retention (ROUGE-L recall)", "higher is better"),
    ("chunk_FPR_%", "Chunk false-positive rate (%)", "lower is better"),
    ("latency_overhead_ms", "Latency overhead (ms / query)", "lower is better"),
    ("cost_usd_per_1k_queries", "API cost (USD / 1k queries)", "lower is better"),
]


def _style(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(f"{title}\n", loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def plot_comparison(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), facecolor=SURFACE)
    setups = list(df["setup"])
    labels = [SETUP_LABELS.get(s, s) for s in setups]
    colors = [SETUP_COLORS.get(s, "#6250d6") for s in setups]
    for ax, (col, title, sub) in zip(axes.flat, PANELS):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).tolist()
        y = range(len(setups))[::-1]
        ax.barh(list(y), vals, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(list(y), labels)
        span = max(vals) if max(vals) > 0 else 1.0  # all-zero panel: plain 0-1 axis
        for yi, v in zip(y, vals):
            ax.text(v + span * 0.02, yi, f"{v:.3g}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, span * 1.18)
        _style(ax, title, sub)
    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_family(fam: pd.DataFrame, out: Path) -> None:
    if fam.empty:
        return
    pivot = fam.pivot(index="attack_family", columns="setup", values="ASR_%").fillna(0)
    order = [s for s in SETUP_COLORS if s in pivot.columns]
    pivot = pivot[order]
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(pivot)), 4.8), facecolor=SURFACE)
    n = len(order)
    width = 0.8 / n
    for i, s in enumerate(order):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(pivot))]
        ax.bar(xs, pivot[s], width=width * 0.92, color=SETUP_COLORS[s], label=SETUP_LABELS[s],
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(len(pivot)), pivot.index, rotation=25, ha="right")
    ax.set_ylabel("ASR (%)", color=INK_2)
    ax.set_ylim(0, 100)
    _style(ax, "Attack success rate by payload family", "lower is better; held-out families never seen in training")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, ncol=min(n, 6), loc="upper left", bbox_to_anchor=(0, -0.28))
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_all(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    figs = results_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plot_comparison(pd.read_csv(results_dir / "comparison.csv"), figs / "comparison.png")
    fam_csv = results_dir / "asr_by_family.csv"
    if fam_csv.exists() and fam_csv.stat().st_size > 1:
        plot_family(pd.read_csv(fam_csv), figs / "asr_by_family.png")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    plot_all(Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results")
    print("figures written to results/figures/")
