"""Threshold sensitivity of the Layer 1 detector (validation selects, test sets only evaluate).

Why the whole score range: two thresholds with no validation score between them make exactly the
same decisions on every validation chunk, so their recall and FPR are identical. When the detector
separates the classes well, a sweep over 0.10-0.25 alone is therefore a flat line. This script
  1. scores the validation set with the checkpoint (or reads ready-made scores, --validation-scores);
  2. sweeps the threshold over the whole score range on a log-odds axis, so the curves show where
     recall and FPR change, and finds the validation-optimal interval (lo, hi]: every threshold in it
     has the highest validation recall under FPR <= 2% and, among those, the lowest FPR;
  3. reports the requested thresholds 0.10 ... 0.25 with their margins to both ends of that interval
     (how far a threshold can move before any validation decision changes);
  4. selects tau* among the requested thresholds on validation only:
       highest recall with FPR <= 2%; ties -> lower FPR; ties -> closest to the midpoint of (lo, hi]
     (the midpoint is the threshold farthest from the nearest benign and the nearest injected score);
  5. only then evaluates the thresholds on the regular and challenge test sets: the saved predictions
     from training, or --rescore-tests to score both sets with the chosen checkpoint (needed for an
     archived checkpoint whose saved predictions were overwritten by a later run).

Outputs in results/supplementary/:
  validation_scores.csv                 chunk_id, label, family, score        (MATLAB: histogram)
  threshold_sensitivity_validation.csv  requested thresholds with margins      (MATLAB: markers)
  threshold_sweep_validation.csv        full-range sweep on validation         (MATLAB: curves)
  threshold_sweep_tests.csv             full-range sweep on the test sets      (evaluation only)
  threshold_range_validation.csv        validation at thresholds from 1e-6 to 0.999999
  threshold_band.csv                    interval ends, midpoint, tau*, axis range (MATLAB)
  threshold_final_evaluation.csv        requested thresholds, tau* and the midpoint on the test sets
  threshold_family_recall.csv           challenge recall per withheld family at the same thresholds
  threshold_summary.json                all of the above plus the rule and the data sources
  figures_python/fig_threshold_sensitivity.png|pdf  validation curves and validation score histogram
  figures_python/fig_threshold_evaluation.png|pdf   regular and challenge curves (evaluation only)

Usage (Windows, project folder):
  .venv\\Scripts\\python.exe supplementary\\threshold_sensitivity.py
  .venv\\Scripts\\python.exe supplementary\\threshold_sensitivity.py --list-checkpoints
  .venv\\Scripts\\python.exe supplementary\\threshold_sensitivity.py --detector-dir models\\<folder> --rescore-tests
  .venv\\Scripts\\python.exe supplementary\\threshold_sensitivity.py --validation-scores <csv with label and score>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

EPS = 1e-12            # scores are clipped to [EPS, 1 - EPS] before taking log-odds
AXIS_CAP = 18.0        # the log-odds axis never extends beyond +-18 (1.5e-8 ... 1 - 1.5e-8)
STEP = 0.1             # sweep step in log-odds
RANGE_THRESHOLDS = (1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project-root")
    ap.add_argument("--data-dir", help="folder with validation.csv, test.csv, challenge.csv (default: data_100k)")
    ap.add_argument("--validation-csv")
    ap.add_argument("--regular-csv", help="regular-test CSV (to check the detector, or to rescore it)")
    ap.add_argument("--challenge-csv", help="challenge CSV (only with --rescore-tests)")
    ap.add_argument("--detector-dir", help="checkpoint folder (default: newest classifier under models/)")
    ap.add_argument("--validation-scores", help="ready-made validation scores (CSV with label and score columns); "
                                                "the checkpoint is then not needed")
    ap.add_argument("--rescore-tests", action="store_true",
                    help="score the regular and challenge sets with the checkpoint instead of reading saved predictions")
    ap.add_argument("--list-checkpoints", action="store_true", help="list the checkpoints and metrics files, then stop")
    ap.add_argument("--regular-predictions", help="default: results/layer1_predictions.csv")
    ap.add_argument("--challenge-predictions", help="default: results/layer1_challenge_predictions.csv")
    ap.add_argument("--label-column")
    ap.add_argument("--score-column")
    ap.add_argument("--family-column")
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(C.THRESHOLD_GRID))
    ap.add_argument("--fpr-limit", type=float, default=C.FPR_LIMIT)
    ap.add_argument("--current", type=float, default=C.CURRENT_THRESHOLD)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- threshold arithmetic
def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def decades(a: float, b: float) -> float:
    """log10 of the odds ratio between b and a: the number of ten-fold steps in odds from a up to b."""
    return (logit(b) - logit(a)) / math.log(10)


class Counts:
    """Confusion counts at any threshold, from sorted scores; a chunk is flagged when score >= threshold."""

    def __init__(self, labels: list[int], scores: list[float]):
        self.pos = sorted(s for y, s in zip(labels, scores) if y == 1)
        self.neg = sorted(s for y, s in zip(labels, scores) if y == 0)

    def at(self, t: float) -> dict:
        tp = len(self.pos) - bisect_left(self.pos, t)
        fp = len(self.neg) - bisect_left(self.neg, t)
        return {"TP": tp, "FN": len(self.pos) - tp, "FP": fp, "TN": len(self.neg) - fp}

    def row(self, t: float, fpr_limit: float) -> dict:
        r = C.rates(self.at(t))
        r["threshold"] = t
        r["feasible"] = int(r["FP"] <= fpr_limit * r["negatives"] + 1e-9)
        return r


def optimal_interval(cn: Counts, fpr_limit: float) -> dict | None:
    """(lo, hi]: the thresholds with the highest recall subject to FPR <= limit and, among them, the
    lowest FPR. No validation score lies strictly inside, so every threshold in it makes the same
    decision on every validation chunk; lo is a benign score and hi an injected score."""
    neg, pos = cn.neg, cn.pos
    if not neg or not pos:
        return None
    allowed = math.floor(fpr_limit * len(neg) + 1e-9)              # false positives the limit allows
    floor_score = neg[len(neg) - 1 - allowed] if allowed < len(neg) else -math.inf
    j = bisect_right(pos, floor_score)                              # first injected score above it
    if j == len(pos):
        return None
    hi = pos[j]
    i = bisect_left(neg, hi) - 1                                    # highest benign score below hi
    lo = neg[i] if i >= 0 else 0.0
    c = cn.at(hi)
    return {"lo": lo, "hi": hi, "midpoint": (lo + hi) / 2, "TP": c["TP"], "FP": c["FP"],
            "recall": c["TP"] / len(pos), "fpr": c["FP"] / len(neg), "allowed_fp": allowed,
            "fpr_edge": floor_score if allowed < len(neg) else None,     # FPR exceeds the limit for tau <= this score
            "separable": neg[-1] < pos[0], "width_decades": decades(lo, hi)}


def select_threshold(grid: list[dict], midpoint: float | None) -> tuple[float | None, list[float]]:
    """Requested thresholds only: highest validation recall with FPR <= limit; ties -> lower FPR;
    remaining ties -> closest to the midpoint of the validation-optimal interval (then the lower one)."""
    feasible = [r for r in grid if r["feasible"]]
    if not feasible:
        return None, []
    best_tp = max(r["TP"] for r in feasible)
    tied = [r for r in feasible if r["TP"] == best_tp]
    best_fp = min(r["FP"] for r in tied)
    tied = sorted(r["threshold"] for r in tied if r["FP"] == best_fp)
    target = midpoint if midpoint is not None else (tied[0] + tied[-1]) / 2
    return min(tied, key=lambda t: (abs(t - target), t)), tied


def axis_range(*score_lists: list[float]) -> tuple[float, float]:
    """Log-odds axis wide enough to show both curves leave their plateaus (0.1st to 99.9th percentile)."""
    xs = sorted(logit(s) for scores in score_lists for s in scores)
    lo = math.floor(C.percentile(xs, 0.1)) - 1
    hi = math.ceil(C.percentile(xs, 99.9)) + 1
    return max(-AXIS_CAP, min(-6.0, lo)), min(AXIS_CAP, max(6.0, hi))


def sweep_thresholds(x_lo: float, x_hi: float, extra: list[float]) -> list[float]:
    n = int(round((x_hi - x_lo) / STEP))
    ts = {expit(x_lo + k * (x_hi - x_lo) / n) for k in range(n + 1)}
    t_lo, t_hi = expit(x_lo), expit(x_hi)
    ts |= {t for t in extra if t is not None and t_lo <= t <= t_hi}
    return sorted(ts)


# --------------------------------------------------------------------------- scores
def scored(detector: C.Detector, rows: list[dict], stem: Path, signature: str, args) -> list[dict]:
    """Scores of a partition, cached next to the other outputs (reused only for the same checkpoint,
    window settings and data)."""
    cols = list(rows[0].keys())
    tc = C.pick_column(cols, C.TEXT_CANDIDATES, what="text", exclude=r"clean|reference|payload")
    lc = C.pick_column(cols, C.LABEL_CANDIDATES, args.label_column, what="label", exclude=r"pred")
    fc = C.pick_column(cols, C.FAMILY_CANDIDATES, args.family_column, what="family", required=False)
    ic = C.pick_column(cols, C.ID_CANDIDATES, required=False)
    cache, meta_file = stem.with_suffix(".csv"), stem.with_suffix(".meta.json")
    key = {"checkpoint": signature, "windows": [detector.max_length, detector.stride], "format": 2,   # 2: 12-digit scores
           "rows": len(rows), "first": rows[0].get(ic, ""), "last": rows[-1].get(ic, "")}
    if cache.exists() and meta_file.exists() and json.loads(meta_file.read_text()).get("key") == key:
        print(f"Scores: reusing {cache.name} (same checkpoint, windows and data)")
        return [{"chunk_id": r["chunk_id"], "label": int(r["label"]), "family": r["family"], "score": float(r["score"])}
                for r in C.read_rows(cache)]
    print(f"Scoring {len(rows):,} chunks for {cache.name} ...")
    start = time.perf_counter()
    scores = detector.score([r[tc] for r in rows], "chunks")
    out = []
    for i, (r, s) in enumerate(zip(rows, scores)):
        y = C.to_label(r[lc])
        out.append({"chunk_id": r.get(ic, str(i)) if ic else str(i), "label": y,
                    "family": C.family_of(r, fc) if y == 1 else "", "score": s})
    C.write_rows(cache, out, ["chunk_id", "label", "family", "score"])
    meta_file.write_text(json.dumps({"key": key, "seconds": round(time.perf_counter() - start, 1)}, indent=2))
    return out


def window_detector(out: Path, model_dir: Path, batch_size: int) -> C.Detector:
    """Detector with the window settings verified earlier for this checkpoint, else the stored ones."""
    det = C.Detector(model_dir, batch_size=batch_size)
    settings = out / "scorer_settings.json"
    if settings.exists():
        s = json.loads(settings.read_text())
        if s.get("signature") == C.checkpoint_signature(model_dir):
            det.max_length, det.stride = s["max_length"], s["stride"]
    return det


def optional_checkpoint(root: Path, explicit: str | None) -> Path | None:
    try:
        return C.find_checkpoint(root, explicit)
    except SystemExit:
        return None


def first_value(obj, keys):
    """First scalar stored under any of `keys` anywhere in a JSON object (depth first)."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and not isinstance(obj[k], (dict, list)):
                return obj[k]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            found = first_value(v, keys)
            if found is not None:
                return found
    return None


def list_checkpoints(root: Path) -> None:
    """Every classifier checkpoint under models/ (archives included) and every metrics file in results/."""
    batch_keys = ("per_device_train_batch_size", "train_batch_size", "batch_size")
    epoch_keys = ("selected_epoch", "best_epoch", "epoch")
    found = []
    for cfg in (root / "models").rglob("config.json") if (root / "models").is_dir() else []:
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        weights = [cfg.parent / w for w in ("model.safetensors", "pytorch_model.bin") if (cfg.parent / w).exists()]
        if weights and any("SequenceClassification" in a for a in data.get("architectures") or []):
            found.append((max(w.stat().st_mtime for w in weights), cfg.parent))
    print("Detector checkpoints (newest first):")
    for mtime, folder in sorted(found, reverse=True):
        info = {}
        for f in sorted(folder.glob("*.json")):
            if f.name in C.HF_FILES:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for name, keys in (("threshold", ("threshold",)), ("batch size", batch_keys), ("epoch", epoch_keys)):
                if name not in info and first_value(data, keys) is not None:
                    info[name] = first_value(data, keys)
        details = ", ".join(f"{k} {v}" for k, v in info.items()) or "no project settings file"
        print(f"  {folder.relative_to(root)}   saved {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}   {details}")
    if not found:
        print("  none under models/")
    print("Metrics files in results/ (newest first):")
    files = sorted((root / "results").glob("*metrics*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        chal = data.get("challenge") if isinstance(data, dict) else None
        info = {"threshold": first_value(data, ("threshold", "selected_threshold", "decision_threshold")),
                "batch size": first_value(data, batch_keys), "epoch": first_value(data, epoch_keys),
                "challenge FN": first_value(chal, ("FN", "fn", "false_negatives")) if chal is not None else None}
        details = ", ".join(f"{k} {v}" for k, v in info.items() if v is not None)
        print(f"  {f.relative_to(root)}   {time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))}   {details}")
    print("Pass the checkpoint that selected the threshold with --detector-dir. If its saved test predictions were "
          "replaced by a later run, add --rescore-tests.")


# --------------------------------------------------------------------------- figures
def tfmt(t: float) -> str:
    """0.1 -> 0.10, 0.125 -> 0.125 (thresholds as written in the text)."""
    return f"{t:.2f}" if abs(t * 100 - round(t * 100)) < 1e-9 else f"{t:g}"


def tick_marks(x_lo: float, x_hi: float) -> tuple[list[float], list[str]]:
    marks = [(1e-8, r"$\mathregular{10^{-8}}$"), (1e-6, r"$\mathregular{10^{-6}}$"), (1e-4, r"$\mathregular{10^{-4}}$"),
             (0.01, "0.01"), (0.5, "0.5"), (0.99, "0.99"), (1 - 1e-4, r"$\mathregular{1{-}10^{-4}}$"),
             (1 - 1e-6, r"$\mathregular{1{-}10^{-6}}$"), (1 - 1e-8, r"$\mathregular{1{-}10^{-8}}$")]
    keep = [(logit(t), lab) for t, lab in marks if x_lo <= logit(t) <= x_hi]
    return [k for k, _ in keep], [lab for _, lab in keep]


def threshold_axes(P, height: float, ratios: list[float]):
    """Two stacked panels sharing a log-odds threshold axis, with room for a legend above them."""
    fig, axes = P.plt.subplots(2, 1, figsize=(P.WIDTH, height), sharex=True,
                               gridspec_kw={"height_ratios": ratios, "hspace": 0.1})
    fig.subplots_adjust(top=1 - 0.5 / height)
    return fig, axes


def decorate(fig, axes, P, band, requested, current, x_lo, x_hi, handles) -> None:
    """Shading (validation-optimal interval, requested range), threshold lines and labels, ticks, legend."""
    from matplotlib.patches import Patch

    show_mid = bool(band) and abs(logit(band["midpoint"]) - logit(current)) > 0.25
    for ax in axes:
        if band:
            ax.axvspan(max(x_lo, logit(band["lo"])), min(x_hi, logit(band["hi"])), color=P.AQUA, alpha=0.14, lw=0, zorder=0)
        ax.axvspan(logit(min(requested)), logit(max(requested)), color=P.GRAY, alpha=0.30, lw=0, zorder=0)
        ax.axvline(logit(current), color=P.INK, lw=0.9, ls="--", zorder=4)
        if show_mid:
            ax.axvline(logit(band["midpoint"]), color=P.NAVY, lw=0.9, ls=":", zorder=4)
        P.tidy(ax)
    top = axes[0].get_ylim()[1]
    left = not show_mid or current < band["midpoint"]          # the left label is right-aligned, and vice versa
    axes[0].text(logit(current), top, f"τ = {current:g} " if left else f" τ = {current:g}", ha="right" if left else "left",
                 va="bottom", fontsize=6.4, color=P.INK)
    if show_mid:
        text = f"midpoint {band['midpoint']:.3g}"
        axes[0].text(logit(band["midpoint"]), top, f" {text}" if left else f"{text} ", ha="left" if left else "right",
                     va="bottom", fontsize=6.4, color=P.NAVY)
    ticks, labels = tick_marks(x_lo, x_hi)
    axes[-1].set_xlim(x_lo, x_hi)
    axes[-1].set_xticks(ticks, labels)
    handles = handles + [Patch(color=P.AQUA, alpha=0.14, lw=0, label="validation-optimal interval"),
                         Patch(color=P.GRAY, alpha=0.30, lw=0, label=f"requested {tfmt(min(requested))}–{tfmt(max(requested))}")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, frameon=False, fontsize=6.2,
               handlelength=1.5, columnspacing=1.4, handletextpad=0.5)


def plot_validation(sweep, grid, band, current, fpr_limit, requested, val, x_lo, x_hi, folder: Path) -> Path:
    import plotting as P
    from matplotlib.lines import Line2D

    fig, (a1, a2) = threshold_axes(P, 3.9, [1.5, 1])
    xs = [r["logit"] for r in sweep]
    a1.plot(xs, [100 * r["recall"] for r in sweep], color=P.ORANGE, lw=1.4, zorder=3)
    a1.plot(xs, [100 * r["fpr"] for r in sweep], color=P.BLUE, lw=1.4, zorder=3)
    req = [r for r in grid if r["requested"]]
    for key, color in (("recall", P.ORANGE), ("fpr", P.BLUE)):
        a1.plot([logit(r["threshold"]) for r in req], [100 * r[key] for r in req], "o", ms=2.4, color=color,
                mec="white", mew=0.3, zorder=5)
    a1.axhline(100 * fpr_limit, color=P.INK2, lw=0.8, ls=(0, (1, 1.5)), zorder=2)
    a1.text(x_hi, 100 * fpr_limit + 1.5, f"FPR limit {100 * fpr_limit:g}% ", ha="right", va="bottom", fontsize=6.2,
            color=P.INK2)
    a1.set_ylim(-4, 104)
    a1.set_yticks([0, 25, 50, 75, 100])
    a1.set_ylabel("Validation chunks (%)")
    width = 0.5
    nbins = int(math.ceil((x_hi - x_lo) / width))
    peak = 1
    for cls, color, name in ((0, P.BLUE, "benign"), (1, P.ORANGE, "injected")):
        counts = [0] * nbins
        for r in val:
            if r["label"] == cls:
                k = int((min(max(logit(r["score"]), x_lo), x_hi - 1e-9) - x_lo) // width)
                counts[min(k, nbins - 1)] += 1
        peak = max(peak, max(counts))
        a2.bar([x_lo + k * width for k in range(nbins)], [c if c else float("nan") for c in counts], width=width,
               align="edge", color=color, edgecolor="white", linewidth=0.25, alpha=0.9, zorder=3)
        k = max(range(nbins), key=lambda i: counts[i])
        a2.text(min(max(x_lo + (k + 0.5) * width, x_lo + 2), x_hi - 2), counts[k] * 1.7, name, ha="center",
                va="bottom", fontsize=6.4, color=color)
    a2.set_yscale("log")
    a2.set_ylim(0.7, peak * 6)
    a2.set_ylabel("Chunks per bin")
    a2.set_xlabel("Threshold τ, or detector score (log-odds axis)")
    handles = [Line2D([], [], color=P.ORANGE, lw=1.4, label="recall (injected chunks flagged)"),
               Line2D([], [], color=P.BLUE, lw=1.4, label="FPR (benign chunks flagged)")]
    decorate(fig, (a1, a2), P, band, requested, current, x_lo, x_hi, handles)
    return P.save(fig, folder, "fig_threshold_sensitivity")


def plot_evaluation(sweep_tests, final, band, current, fpr_limit, requested, x_lo, x_hi, folder: Path) -> Path:
    import plotting as P
    from matplotlib.lines import Line2D

    fig, (a1, a2) = threshold_axes(P, 3.3, [1.4, 1])
    low, handles = 100.0, []
    for split, color, name in (("regular", P.BLUE, "regular test"), ("challenge", P.ORANGE, "challenge set")):
        rows = [r for r in sweep_tests if r["split"] == split]
        xs = [r["logit"] for r in rows]
        a1.plot(xs, [100 * r["recall"] for r in rows], color=color, lw=1.3, zorder=3)
        a2.plot(xs, [100 * r["fpr"] for r in rows], color=color, lw=1.3, zorder=3)
        low = min([low] + [100 * r["recall"] for r in rows if r["threshold"] <= 0.99])
        req = [r for r in final if r["split"] == split and r["requested"]]
        a1.plot([logit(r["threshold"]) for r in req], [100 * r["recall"] for r in req], "o", ms=2.4, color=color,
                mec="white", mew=0.3, zorder=5)
        handles.append(Line2D([], [], color=color, lw=1.3, label=name))
    floor = max(0.0, math.floor(low - 0.5))
    a1.set_ylim(floor, 100 + (100 - floor) * 0.06)
    a1.set_ylabel("Recall (%)")
    a2.axhline(100 * fpr_limit, color=P.INK2, lw=0.8, ls=(0, (1, 1.5)), zorder=2)
    a2.text(x_hi, 100 * fpr_limit, f"{100 * fpr_limit:g}% (validation limit) ", ha="right", va="bottom", fontsize=6.0,
            color=P.INK2)
    a2.set_ylim(-0.15, 100 * fpr_limit * 2.5)
    a2.set_ylabel("FPR (%)")
    a2.set_xlabel("Threshold τ (log-odds axis), evaluation only")
    decorate(fig, (a1, a2), P, band, requested, current, x_lo, x_hi, handles)
    return P.save(fig, folder, "fig_threshold_evaluation")


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    args = parse_args(argv)
    root = C.find_project_root(args.project_root)
    if args.list_checkpoints:
        list_checkpoints(root)
        return 0
    out = C.out_dir(root, args.out_dir)
    requested = sorted({round(t, 6) for t in args.thresholds})
    current = round(args.current, 6)
    need_model = args.validation_scores is None or args.rescore_tests
    model_dir = C.find_checkpoint(root, args.detector_dir) if need_model else optional_checkpoint(root, args.detector_dir)
    stored = C.checkpoint_settings(model_dir).get("threshold") if model_dir else None

    # test scores: used only for the final evaluation, after the selection
    pred_paths = {"regular": Path(args.regular_predictions) if args.regular_predictions else root / "results" / "layer1_predictions.csv",
                  "challenge": Path(args.challenge_predictions) if args.challenge_predictions else root / "results" / "layer1_challenge_predictions.csv"}
    tests, test_cols = {}, {}
    if not args.rescore_tests:
        for split, path in pred_paths.items():
            if not path.exists():
                raise SystemExit(f"Saved predictions not found: {path}. Pass --{split}-predictions, or --rescore-tests "
                                 f"to score the test sets with the checkpoint.")
            tests[split], test_cols[split] = C.load_saved_predictions(path, args.label_column, args.score_column,
                                                                      args.family_column)

    detector, scorer, parts = None, {"verification": {"status": "not needed (validation scores read from a file)"}}, {}
    if need_model:
        data_dir = Path(args.data_dir) if args.data_dir else C.default_data_dir(root)
        wanted = (("validation",) if args.validation_scores is None else ()) + ("regular",) + \
                 (("challenge",) if args.rescore_tests else ())
        print("Benchmark partitions:")
        parts = C.load_partitions(data_dir, wanted, {"validation": args.validation_csv, "regular": args.regular_csv,
                                                     "challenge": args.challenge_csv})
        if args.rescore_tests:
            detector = window_detector(out, model_dir, args.batch_size)
            scorer = {"signature": C.checkpoint_signature(model_dir),
                      "verification": {"status": "not compared with saved predictions (test sets rescored with this checkpoint)"}}
            print(f"Detector: {detector.description}")
        else:
            detector, scorer = C.aligned_detector(root, out, model_dir, tests["regular"], parts["regular"], args.batch_size)

    # validation scores: the only data that influences the choice
    if args.validation_scores:
        val, val_cols = C.load_saved_predictions(Path(args.validation_scores), args.label_column, args.score_column,
                                                 args.family_column)
        val_source = f"read from {Path(args.validation_scores).name}"
        C.write_rows(out / "validation_scores.csv", val, ["chunk_id", "label", "family", "score"])
        (out / "validation_scores.meta.json").unlink(missing_ok=True)       # not a cache of this checkpoint
    else:
        val = scored(detector, parts["validation"], out / "validation_scores", scorer["signature"], args)
        val_source, val_cols = "scored with the checkpoint", {}
    if args.rescore_tests:
        for split in ("regular", "challenge"):
            tests[split] = scored(detector, parts[split], out / f"rescored_{split}", scorer["signature"], args)
            test_cols[split] = {"label": "label", "score": "score", "family": "family", "id": "chunk_id"}

    labels, scores = [r["label"] for r in val], [r["score"] for r in val]
    if not 0 < sum(labels) < len(labels):
        raise SystemExit("The validation scores need both benign and injected chunks.")
    cn = Counts(labels, scores)
    band = optimal_interval(cn, args.fpr_limit)
    mid = band["midpoint"] if band else None

    # requested thresholds on validation, with their margins to both ends of the optimal interval
    grid = []
    for t in sorted(set(requested) | {current}):
        r = cn.row(t, args.fpr_limit)
        r["requested"] = int(t in requested)
        r["current"] = int(abs(t - current) < 1e-9)
        r["lower_margin"] = decades(band["lo"], t) if band else float("nan")
        r["upper_margin"] = decades(t, band["hi"]) if band else float("nan")
        r["in_optimal_interval"] = int(bool(band) and band["lo"] < t <= band["hi"])
        grid.append(r)
    tau_star, tied = select_threshold([r for r in grid if r["requested"]], mid)
    for r in grid:
        r["selected"] = int(r["threshold"] == tau_star)
        r["tied"] = int(r["threshold"] in tied)
    range_rows = [cn.row(t, args.fpr_limit) for t in sorted(set(RANGE_THRESHOLDS) | {current})]
    for r in range_rows:
        r["current"] = int(abs(r["threshold"] - current) < 1e-9)

    print(f"\nValidation ({len(val):,} chunks: {sum(labels):,} injected, {len(labels) - sum(labels):,} benign; {val_source})")
    print(f"  highest benign score {cn.neg[-1]:.6g}, lowest injected score {cn.pos[0]:.6g}")
    if band:
        print(f"  validation-optimal interval ({band['lo']:.6g}, {band['hi']:.6g}]: recall {C.pct(band['recall'])}, "
              f"FPR {C.pct(band['fpr'])}; midpoint {band['midpoint']:.4g}")
    print(f"  {'tau':>6} {'recall':>8} {'FPR':>7}  FPR<=2%  margin below / above (decades of odds)")
    for r in grid:
        print(f"  {r['threshold']:>6g} {C.pct(r['recall']):>8} {C.pct(r['fpr']):>7}  {'yes' if r['feasible'] else 'no ':<7}"
              f"  {r['lower_margin']:+.2f} / {r['upper_margin']:+.2f}"
              f"{'  <- selected' if r['selected'] else ''}{'  (current)' if r['current'] else ''}")
    print(f"Selected among the requested thresholds: tau* = {tau_star:g}" if tau_star is not None else
          "No requested threshold meets the FPR limit on validation.")
    if stored is not None and abs(float(stored) - current) > 1e-6:
        print(f"NOTE: this checkpoint ({model_dir.name}) stores threshold {stored}, not {current:g}. If {current:g} was "
              f"selected for another training run, analyse that checkpoint (--list-checkpoints).")

    # full-range sweeps (validation for the figure; the test sets for the evaluation figure only)
    x_lo, x_hi = axis_range(scores, *[[r["score"] for r in tests[s]] for s in tests])
    extra = requested + [current, mid]
    if band:
        extra += [band["lo"], math.nextafter(band["lo"], 1), band["hi"], math.nextafter(band["hi"], 1)]
    ts = sweep_thresholds(x_lo, x_hi, extra)

    def sweep_row(counts: Counts, t: float) -> dict:
        c = counts.at(t)
        return {"threshold": t, "logit": logit(t), **c, "recall": c["TP"] / max(1, c["TP"] + c["FN"]),
                "fpr": c["FP"] / max(1, c["FP"] + c["TN"])}

    sweep = [sweep_row(cn, t) for t in ts]
    test_counts = {s: Counts([r["label"] for r in tests[s]], [r["score"] for r in tests[s]]) for s in ("regular", "challenge")}
    sweep_tests = [{"split": s, **sweep_row(test_counts[s], t)} for s in ("regular", "challenge") for t in ts]

    # final evaluation on the test sets (not used for the choice)
    roles: dict[float, list[str]] = {}
    for t in requested:
        roles.setdefault(t, []).append("requested")
    roles.setdefault(current, []).append("current")
    if tau_star is not None:
        roles.setdefault(tau_star, []).append("selected")
    if mid is not None:
        roles.setdefault(mid, []).append("midpoint")
    final = []
    for split in ("regular", "challenge"):
        for t in sorted(roles):
            r = {"split": split, "threshold": t, "role": ", ".join(roles[t]), "requested": int("requested" in roles[t]),
                 **test_counts[split].row(t, args.fpr_limit)}
            final.append(r)
    family_rows, fam = [], {}
    for r in tests["challenge"]:
        if r["label"] == 1:
            fam.setdefault(r["family"] or "(no family column)", []).append(r["score"])
    for name in sorted(fam):
        for t in sorted(roles):
            hit = sum(s >= t for s in fam[name])
            lo, hi = C.wilson(hit, len(fam[name]))
            family_rows.append({"family": name, "threshold": t, "role": ", ".join(roles[t]), "n": len(fam[name]),
                                "detected": hit, "missed": len(fam[name]) - hit, "recall": hit / len(fam[name]),
                                "recall_lo": lo, "recall_hi": hi})
    print("\nFinal evaluation on the test sets (not used for the choice):")
    for r in final:
        print(f"  {r['split']:<9} tau={r['threshold']:<8.4g} ({r['role']}): recall {C.pct(r['recall'])}, FPR {C.pct(r['fpr'])}, "
              f"FN {r['FN']}, FP {r['FP']}")

    # files
    C.write_rows(out / "threshold_sensitivity_validation.csv", grid,
                 ["threshold", "TP", "FN", "FP", "TN", "positives", "negatives", "recall", "recall_lo", "recall_hi", "fpr",
                  "fpr_lo", "fpr_hi", "precision", "accuracy", "f1", "feasible", "tied", "selected", "current", "requested",
                  "lower_margin", "upper_margin", "in_optimal_interval"])
    C.write_rows(out / "threshold_sweep_validation.csv", sweep, ["threshold", "logit", "TP", "FN", "FP", "TN", "recall", "fpr"])
    C.write_rows(out / "threshold_sweep_tests.csv", sweep_tests,
                 ["split", "threshold", "logit", "TP", "FN", "FP", "TN", "recall", "fpr"])
    C.write_rows(out / "threshold_range_validation.csv", range_rows,
                 ["threshold", "TP", "FN", "FP", "TN", "recall", "fpr", "feasible", "current"])
    band_rows = [("current", current), ("selected", tau_star), ("fpr_limit", args.fpr_limit),
                 ("requested_min", min(requested)), ("requested_max", max(requested)), ("axis_min", x_lo), ("axis_max", x_hi),
                 ("benign_max", cn.neg[-1]), ("injected_min", cn.pos[0])]
    if band:
        band_rows += [("interval_lo", band["lo"]), ("interval_hi", band["hi"]), ("midpoint", band["midpoint"]),
                      ("interval_recall", band["recall"]), ("interval_fpr", band["fpr"])]
    with open(out / "threshold_band.csv", "w", encoding="utf-8") as fh:
        fh.write("name,value\n" + "".join(f"{k},{'' if v is None else repr(float(v))}\n" for k, v in band_rows))
    C.write_rows(out / "threshold_final_evaluation.csv", final,
                 ["split", "threshold", "role", "requested", "TP", "FN", "FP", "TN", "accuracy", "accuracy_lo", "accuracy_hi",
                  "precision", "recall", "recall_lo", "recall_hi", "fpr", "fpr_lo", "fpr_hi", "f1"])
    C.write_rows(out / "threshold_family_recall.csv", family_rows,
                 ["family", "threshold", "role", "n", "detected", "missed", "recall", "recall_lo", "recall_hi"])
    fig_val = plot_validation(sweep, grid, band, current, args.fpr_limit, requested, val, x_lo, x_hi, out / "figures_python")
    fig_eval = plot_evaluation(sweep_tests, final, band, current, args.fpr_limit, requested, x_lo, x_hi, out / "figures_python")

    cur_row = next(r for r in grid if r["current"])
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector_dir": str(model_dir) if model_dir else None,
        "detector": detector.description if detector else (model_dir.name if model_dir else "not loaded"),
        "checkpoint_threshold": stored,
        "checkpoint_threshold_differs": stored is not None and abs(float(stored) - current) > 1e-6,
        "scorer_check": scorer["verification"],
        "validation": {"chunks": len(val), "injected": sum(labels), "benign": len(labels) - sum(labels), "source": val_source,
                       "columns": val_cols, "benign_max": cn.neg[-1], "injected_min": cn.pos[0],
                       "benign_median": C.percentile(cn.neg, 50), "injected_median": C.percentile(cn.pos, 50),
                       "roc_auc": C.roc_auc(labels, scores)},
        "test_scores": "rescored with the checkpoint" if args.rescore_tests else "saved predictions from training",
        "thresholds": requested, "fpr_limit": args.fpr_limit, "current_threshold": current,
        "rule": ("highest validation recall among the requested thresholds with validation FPR <= limit; ties: lower "
                 "FPR, then the threshold closest to the midpoint of the validation-optimal interval"),
        "optimal_interval": band,
        "current": {"threshold": current, "in_optimal_interval": bool(cur_row["in_optimal_interval"]),
                    "lower_margin": cur_row["lower_margin"], "upper_margin": cur_row["upper_margin"],
                    "recall": cur_row["recall"], "fpr": cur_row["fpr"]},
        "feasible_thresholds": [r["threshold"] for r in grid if r["requested"] and r["feasible"]],
        "tied_thresholds": tied, "selected_threshold": tau_star,
        "selected_equals_current": tau_star is not None and abs(tau_star - current) < 1e-9,
        "axis": [x_lo, x_hi], "grid": grid, "range": range_rows, "final_evaluation": final, "family_recall": family_rows,
        "test_sources": ({k: f"rescored_{k}.csv" for k in pred_paths} if args.rescore_tests
                         else {k: str(v) for k, v in pred_paths.items()}),
        "test_columns": test_cols, "figures": [str(fig_val), str(fig_eval)],
    }
    (out / "threshold_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote validation_scores.csv, threshold_*.csv, threshold_summary.json, figures_python/{fig_val.name} and "
          f"figures_python/{fig_eval.name} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
