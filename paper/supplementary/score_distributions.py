"""Score distributions of benign and injected chunks, from the saved per-chunk predictions.

Reads results/layer1_predictions.csv (regular test) and results/layer1_challenge_predictions.csv
(challenge), as written by train_layer1.py, and draws the scores of benign and injected chunks as
two separate distributions per test set, with the operating threshold 0.175 as a dashed line.

Outputs in results/supplementary/:
  scores_regular.csv, scores_challenge.csv   label, score, family (MATLAB: plot_score_distributions.m)
  score_distribution_stats.csv               per test set and class: n, mean, median, percentiles, extremes,
                                             chunks on each side of the threshold
  score_distribution_summary.json            the same plus ROC AUC and the separation gap
  figures_python/fig_score_distributions.png|pdf

Usage (Windows, project folder):
  .venv\\Scripts\\python.exe supplementary\\score_distributions.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project-root")
    ap.add_argument("--regular-predictions", help="default: results/layer1_predictions.csv")
    ap.add_argument("--challenge-predictions", help="default: results/layer1_challenge_predictions.csv")
    ap.add_argument("--label-column")
    ap.add_argument("--score-column")
    ap.add_argument("--family-column")
    ap.add_argument("--threshold", type=float, default=C.CURRENT_THRESHOLD)
    ap.add_argument("--bin-width", type=float, default=0.025, help="histogram bin width (edges align with 0.175)")
    ap.add_argument("--out-dir")
    return ap.parse_args(argv)


def class_stats(values: list[float], threshold: float) -> dict:
    return {"n": len(values), "mean": statistics.fmean(values) if values else float("nan"),
            "median": C.percentile(values, 50), "p01": C.percentile(values, 1), "p05": C.percentile(values, 5),
            "p95": C.percentile(values, 95), "p99": C.percentile(values, 99),
            "min": min(values) if values else float("nan"), "max": max(values) if values else float("nan"),
            "at_or_above_threshold": sum(v >= threshold for v in values),
            "below_threshold": sum(v < threshold for v in values)}


def plot(data: dict, threshold: float, bin_width: float, folder: Path) -> Path:
    import numpy as np
    import plotting as P

    edges = np.round(np.arange(0, 1 + bin_width / 2, bin_width), 6)
    fig, axes = P.plt.subplots(2, 2, figsize=(P.WIDTH, 2.9), sharex=True, gridspec_kw={"hspace": 0.28, "wspace": 0.12})
    titles = {"regular": "Regular test", "challenge": "Challenge set"}
    peak = 1
    for col, split in enumerate(("regular", "challenge")):
        for row, (cls, name, color) in enumerate(((0, "Benign", P.BLUE), (1, "Injected", P.ORANGE))):
            ax = axes[row, col]
            vals = np.array([r["score"] for r in data[split] if r["label"] == cls])
            counts, _ = np.histogram(np.clip(vals, 0, 1), bins=edges)
            peak = max(peak, counts.max(initial=0))
            heights = np.where(counts > 0, counts, np.nan)
            ax.bar(edges[:-1], heights, width=bin_width, align="edge", color=color, edgecolor="white", linewidth=0.3)
            ax.axvline(threshold, color=P.INK, lw=0.9, ls="--")
            ax.set_yscale("log")
            if cls == 0:
                side = int((vals >= threshold).sum())
                note = f"{side:,} of {len(vals):,} ≥ τ"
            else:
                side = int((vals < threshold).sum())
                note = f"{side:,} of {len(vals):,} < τ"
            ax.text(0.97, 0.93, note, transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=P.INK,
                    bbox={"fc": "white", "ec": "none", "pad": 0.8, "alpha": 0.85})
            if row == 0:
                ax.set_title(titles[split], color=P.INK, pad=3)
            if col == 0:
                ax.set_ylabel(f"{name} chunks")
            if row == 1:
                ax.set_xlabel("Detector score s(c)")
            P.tidy(ax)
    for ax in axes.flat:
        ax.set_ylim(0.7, peak * 2.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, threshold, 0.5, 1.0], ["0", "τ", "0.5", "1"])
    return P.save(fig, folder, "fig_score_distributions")


def main(argv=None) -> int:
    args = parse_args(argv)
    root = C.find_project_root(args.project_root)
    out = C.out_dir(root, args.out_dir)
    paths = {"regular": Path(args.regular_predictions) if args.regular_predictions else root / "results" / "layer1_predictions.csv",
             "challenge": Path(args.challenge_predictions) if args.challenge_predictions else root / "results" / "layer1_challenge_predictions.csv"}
    data, columns, stats_rows, summary = {}, {}, [], {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "threshold": args.threshold, "bin_width": args.bin_width, "splits": {}}
    for split, path in paths.items():
        if not path.exists():
            raise SystemExit(f"Saved predictions not found: {path} (pass --{split}-predictions).")
        data[split], columns[split] = C.load_saved_predictions(path, args.label_column, args.score_column, args.family_column)
        C.write_rows(out / f"scores_{split}.csv", data[split], ["label", "score", "family"])
        benign = [r["score"] for r in data[split] if r["label"] == 0]
        injected = [r["score"] for r in data[split] if r["label"] == 1]
        sb, si = class_stats(benign, args.threshold), class_stats(injected, args.threshold)
        misses = sorted(v for v in injected if v < args.threshold)
        info = {
            "source": str(path), "columns": columns[split], "benign": sb, "injected": si,
            "roc_auc": C.roc_auc([r["label"] for r in data[split]], [r["score"] for r in data[split]]),
            "separation_gap": (si["min"] - sb["max"]) if benign and injected else float("nan"),
            "false_alarms": sb["at_or_above_threshold"], "misses": si["below_threshold"],
            "median_score_of_misses": C.percentile(misses, 50) if misses else float("nan"),
            "max_score_of_misses": max(misses) if misses else float("nan"),
            "misses_between_0.10_and_threshold": sum(0.10 <= v < args.threshold for v in misses),
            "benign_between_0.10_and_threshold": sum(0.10 <= v < args.threshold for v in benign),
        }
        summary["splits"][split] = info
        for cls, st in (("benign", sb), ("injected", si)):
            stats_rows.append({"split": split, "class": cls, **st})
        print(f"{split:<9} benign n={sb['n']:,} max={sb['max']:.4f} | injected n={si['n']:,} min={si['min']:.4f} | "
              f"false alarms {info['false_alarms']}, misses {info['misses']}, AUC {info['roc_auc']:.4f}")
    C.write_rows(out / "score_distribution_stats.csv", stats_rows,
                 ["split", "class", "n", "mean", "median", "p01", "p05", "p95", "p99", "min", "max",
                  "at_or_above_threshold", "below_threshold"])
    figure = plot(data, args.threshold, args.bin_width, out / "figures_python")
    summary["figure"] = str(figure)
    (out / "score_distribution_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote scores_regular.csv, scores_challenge.csv, score_distribution_stats.csv, "
          f"score_distribution_summary.json and {figure.name} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
