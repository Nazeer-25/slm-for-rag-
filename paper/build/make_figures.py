"""Print-ready figures for the paper (IEEE single column, 3.45 in wide).

Reads numbers.json (written by paper_numbers.py) so every figure agrees with the
tables. Writes 600-dpi PNG (embedded in the .docx) and vector PDF/SVG copies.
Usage:  python paper_numbers.py && python make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
N = json.loads((HERE / "numbers.json").read_text())

# Palette: validated categorical slots 1-2 on white + neutral ink/chrome.
BLUE, ORANGE = "#2a78d6", "#eb6834"          # regular test, challenge
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
GRAY_LIGHT, GRAY_MID, GRAY_DARK = "#e6e5df", "#c9c7bf", "#8f8d86"
BLUE_TINT, ORANGE_TINT = "#e3eefb", "#fde8df"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

W = 3.45
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def save(fig, name):
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}%"


# --------------------------------------------------------------------------
# Fig. 1 - architecture (detection-centred)
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, title, sub=None, fc="white", ec=BASE, lw=0.8, bold=False, tsize=7.2, ssize=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=tsize,
                fontweight="bold" if bold else "normal", color=INK)
        ax.text(x + w / 2, y + h * 0.29, sub, ha="center", va="center", fontsize=ssize, color=INK2)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize,
                fontweight="bold" if bold else "normal", color=INK)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7,
                                shrinkA=0, shrinkB=0))


def fig_architecture():
    H = 3.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Row 1: retrieval (frozen, untrusted output)
    y1, h1 = 2.55, 0.42
    box(ax, 0.02, y1, 0.86, h1, "User question", "trusted")
    box(ax, 1.15, y1, 0.98, h1, "MiniLM encoder", "frozen, 22.7M")
    box(ax, 2.40, y1, 1.03, h1, "Chroma top-k", "retrieved chunks")
    arrow(ax, 0.88, y1 + h1 / 2, 1.15, y1 + h1 / 2)
    arrow(ax, 2.13, y1 + h1 / 2, 2.40, y1 + h1 / 2)
    ax.text(2.915, y1 - 0.09, "untrusted", ha="center", va="center", fontsize=6.2,
            color=INK2, style="italic")

    # Row 2: Layer 1 detector (the trained module)
    y2, h2 = 1.48, 0.78
    ax.add_patch(FancyBboxPatch((0.02, y2), W - 0.04, h2, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=BLUE_TINT, ec=BLUE, lw=1.1))
    ax.text(0.12, y2 + h2 - 0.15, "Layer 1: DistilBERT injection detector", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=INK)
    ax.text(W - 0.12, y2 + h2 - 0.15, "trained: 7.68M of 66.96M params", ha="right", va="center",
            fontsize=6.2, color=INK2)
    # three mini steps inside the detector
    sy, sh = y2 + 0.08, 0.42
    box(ax, 0.12, sy, 0.98, sh, "Windowing", "256 tok, stride 64", fc="white", ec=BLUE, lw=0.7, tsize=6.8, ssize=6.0)
    box(ax, 1.25, sy, 0.98, sh, "Classify window", "p(injection)", fc="white", ec=BLUE, lw=0.7, tsize=6.8, ssize=6.0)
    box(ax, 2.38, sy, 0.95, sh, "Chunk score", "s = max p", fc="white", ec=BLUE, lw=0.7, tsize=6.8, ssize=6.0)
    arrow(ax, 1.10, sy + sh / 2, 1.25, sy + sh / 2, color=BLUE)
    arrow(ax, 2.23, sy + sh / 2, 2.38, sy + sh / 2, color=BLUE)
    arrow(ax, 2.915, y1, 2.915, y2 + h2, color=INK2)

    # Row 3: routing
    y3, h3 = 0.66, 0.46
    ax.text(0.72, (y2 + y3 + h3) / 2 + 0.01, "s < τ: pass unchanged", ha="center", va="center",
            fontsize=6.2, color=INK2, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.text(2.62, (y2 + y3 + h3) / 2 + 0.01, "s ≥ τ = 0.175: flagged", ha="center", va="center",
            fontsize=6.2, color=INK2, bbox=dict(fc="white", ec="none", pad=0.6))
    box(ax, 1.94, y3, 1.49, h3, "Layer 2: clause sanitizer", "spaCy + rules, no LLM call",
        fc=ORANGE_TINT, ec=ORANGE, lw=0.9, tsize=6.9, ssize=6.0)
    box(ax, 0.02, y3, 1.40, h3, "Context assembly", "untrusted-data reminder", tsize=6.9, ssize=6.0)
    arrow(ax, 0.72, y2, 0.72, y3 + h3)
    arrow(ax, 2.62, y2, 2.62, y3 + h3)
    arrow(ax, 1.94, y3 + h3 / 2, 1.42, y3 + h3 / 2)

    # Row 4: generation
    y4, h4 = 0.02, 0.42
    box(ax, 0.02, y4, 1.62, h4, "Local LLM (Ollama)", "llama3.2:3b, frozen", tsize=6.9, ssize=6.0)
    box(ax, 1.94, y4, 1.49, h4, "Answer + trace", "scores, edits, latency", tsize=6.9, ssize=6.0)
    arrow(ax, 0.72, y3, 0.72, y4 + h4)
    arrow(ax, 1.64, y4 + h4 / 2, 1.94, y4 + h4 / 2)
    save(fig, "fig1_architecture")


# --------------------------------------------------------------------------
# Fig. 2 - benchmark partitions (proportional bar)
# --------------------------------------------------------------------------
def fig_partitions():
    rows = [  # label, total, colour, annotation after the bar
        ("Training pool", 70000, GRAY_LIGHT, None),
        ("Validation", 10000, GRAY_MID, "24 seen families; selects epoch and \u03c4"),
        ("Regular test", 10000, BLUE, "24 seen families; 500 new divisions"),
        ("Challenge", 10000, ORANGE, "8 withheld families; 500 new divisions"),
    ]
    fig, ax = plt.subplots(figsize=(W, 1.30))
    ys = list(range(len(rows)))[::-1]
    for y, (label, n, color, note) in zip(ys, rows):
        ax.barh(y, n / 1000, height=0.58, color=color, edgecolor="none")
        if label == "Training pool":
            ax.barh(y, 6, height=0.58, color=GRAY_DARK, edgecolor="none")
            ax.text(3, y, "6k", ha="center", va="center", fontsize=6.6, color="white")
            ax.text(38, y, "70,000 chunks; 24 seen families", ha="center", va="center", fontsize=6.6, color=INK)
        else:
            ax.text(n / 1000 / 2, y, "10k", ha="center", va="center", fontsize=6.6,
                    color="white" if color in (BLUE, ORANGE) else INK)
            ax.text(n / 1000 + 1.2, y, note, ha="left", va="center", fontsize=6.6, color=INK2)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70], ["0", "10k", "20k", "30k", "40k", "50k", "60k", "70k"])
    ax.set_ylim(-0.55, 3.45)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Chunks (each partition 50% benign / 50% poisoned; dark segment = 3k/3k fine-tuning sample)",
                  labelpad=2, fontsize=6.4)
    save(fig, "fig2_partitions")


# --------------------------------------------------------------------------
# Fig. 3 - confusion matrices
# --------------------------------------------------------------------------
def seq_color(v):
    idx = 0 if v == 0 else max(1, min(len(SEQ) - 1, int(round(v * (len(SEQ) - 1)))))
    return SEQ[idx], ("white" if idx >= 4 else INK)


def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.62), gridspec_kw=dict(wspace=0.10))
    for ax, key, title in ((axes[0], "regular", "(a) Regular test"), (axes[1], "challenge", "(b) Challenge")):
        c = N[key]["counts"]
        cells = [[c["TN"], c["FP"]], [c["FN"], c["TP"]]]
        for i in range(2):
            row_total = sum(cells[i])
            for j in range(2):
                v = cells[i][j]
                fc, tc = seq_color(v / row_total)
                ax.add_patch(Rectangle((j + 0.02, 1 - i + 0.02), 0.96, 0.96, fc=fc,
                                       ec=GRID if fc == "#ffffff" else "none", lw=0.6))
                ax.text(j + 0.5, 1 - i + 0.60, f"{v:,}", ha="center", va="center", fontsize=8,
                        fontweight="bold", color=tc)
                ax.text(j + 0.5, 1 - i + 0.33, pct(v / row_total, 1), ha="center", va="center",
                        fontsize=6.6, color=tc)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect("equal")
        ax.set_xticks([0.5, 1.5], ["Benign", "Injected"])
        ax.set_yticks([1.5, 0.5], ["Benign", "Injected"] if key == "regular" else ["", ""])
        ax.tick_params(length=0, pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel("Predicted", labelpad=2)
    axes[0].set_ylabel("Actual", labelpad=2)
    save(fig, "fig3_confusion")


# --------------------------------------------------------------------------
# Fig. 4 - headline metrics with 95 % Wilson intervals
# --------------------------------------------------------------------------
def fig_metrics():
    names = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall (TPR)"),
             ("specificity", "Specificity (1-FPR)")]
    fig, ax = plt.subplots(figsize=(W, 1.55))
    ys = list(range(len(names)))[::-1]
    for key, color, marker, off, label in (("regular", BLUE, "o", 0.14, "Regular test (seen families)"),
                                           ("challenge", ORANGE, "s", -0.14, "Challenge (withheld families)")):
        for y, (m, _) in zip(ys, names):
            v = 100 * N[key][m]
            lo, hi = (100 * x for x in N[key][m + "_ci"])
            ax.plot([lo, hi], [y + off, y + off], color=color, lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot(v, y + off, linestyle="none", marker=marker, ms=4.6, mfc=color, mec="white", mew=1.0,
                    zorder=3, label=label if m == "accuracy" else None)
            if key == "challenge" and v < 99.5:
                ax.text(hi + 0.08, y + off, f"{v:.2f}%", ha="left", va="center", fontsize=6.6, color=INK)
    ax.set_yticks(ys, [n for _, n in names])
    ax.set_xlim(95.0, 100.25)
    ax.set_xticks([95, 96, 97, 98, 99, 100], ["95%", "96%", "97%", "98%", "99%", "100%"])
    ax.set_ylim(-0.55, 3.55)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False, handlelength=1.0,
              columnspacing=1.2, handletextpad=0.4, borderaxespad=0.1)
    ax.text(1.0, -0.30, "x-axis starts at 95%; whiskers are 95% Wilson intervals",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color=MUTED)
    save(fig, "fig4_metrics_ci")


# --------------------------------------------------------------------------
# Fig. 5 - recall per withheld attack family
# --------------------------------------------------------------------------
def fig_families():
    fam = N["families"]
    rows = sorted(fam.items(), key=lambda kv: (kv[1]["recall"], kv[0]))
    fig, ax = plt.subplots(figsize=(W, 1.95))
    ys = list(range(len(rows)))[::-1]
    for y, (name, f) in zip(ys, rows):
        miss = f["missed"]
        color = ORANGE if miss else GRAY_MID
        ax.barh(y, 100 * f["recall"], height=0.62, color=color, edgecolor="none")
        if miss:
            label = f"{100 * f['recall']:.2f}%  ({miss} missed)"
        else:
            label = "100%  (0 missed)"
        ax.text(100 * f["recall"] - 1.2, y, label, ha="right", va="center", fontsize=6.6,
                color="white" if miss else INK)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Recall on the challenge partition (threshold 0.175)", labelpad=2)
    save(fig, "fig5_family_recall")


if __name__ == "__main__":
    fig_architecture()
    fig_partitions()
    fig_confusion()
    fig_metrics()
    fig_families()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
