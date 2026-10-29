"""Threshold sensitivity of the Layer 1 detector (validation selects, test sets only evaluate).

1. Loads the fine-tuned checkpoint and confirms that it reproduces the saved test predictions.
2. Scores every validation chunk (cached in validation_scores.csv).
3. Sweeps the thresholds 0.10, 0.125, 0.15, 0.175, 0.20, 0.225 and 0.25 on the validation set:
   recall and false-positive rate (FPR) with 95% Wilson intervals.
4. Selects tau* on validation only: the highest validation recall among the thresholds with
   validation FPR <= 2%; ties go to the lower FPR, then to the centre of the tied range
   (the plateau-midpoint rule of the training code).
5. Only then evaluates tau* and the current threshold 0.175 on the regular and challenge test
   sets, using the saved per-chunk predictions from training.

Outputs in results/supplementary/:
  validation_scores.csv                 chunk_id, label, family, score (validation set)
  threshold_sensitivity_validation.csv  one row per threshold   (MATLAB: plot_threshold_sensitivity.m)
  threshold_final_evaluation.csv        tau* and 0.175 on the regular and challenge test sets
  threshold_family_recall.csv           challenge recall per withheld family at tau* and 0.175
  threshold_summary.json                selection, rule, data sources, scorer check
  figures_python/fig_threshold_sensitivity.png|pdf

Usage (Windows, project folder):
  .venv\\Scripts\\python.exe supplementary\\threshold_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project-root")
    ap.add_argument("--data-dir", help="folder with validation.csv, test.csv, challenge.csv (default: data_100k)")
    ap.add_argument("--validation-csv")
    ap.add_argument("--regular-csv", help="regular-test CSV (used to check the detector against the saved scores)")
    ap.add_argument("--detector-dir", help="checkpoint folder (default: newest classifier under models/)")
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


def select_threshold(grid: list[dict], fpr_limit: float) -> tuple[float | None, list[float]]:
    """Highest validation recall subject to FPR <= limit; ties: lower FPR, then centre of the tied range."""
    feasible = [r for r in grid if r["FP"] <= fpr_limit * r["negatives"] + 1e-9]
    if not feasible:
        return None, []
    best_tp = max(r["TP"] for r in feasible)
    tied = [r for r in feasible if r["TP"] == best_tp]
    best_fp = min(r["FP"] for r in tied)
    tied = sorted(r["threshold"] for r in tied if r["FP"] == best_fp)
    centre = (tied[0] + tied[-1]) / 2
    return min(tied, key=lambda t: (abs(t - centre), -t)), [r["threshold"] for r in feasible]


def plot(grid: list[dict], tau_star: float | None, current: float, fpr_limit: float, folder: Path) -> Path:
    import plotting as P

    ts = [r["threshold"] for r in grid]
    fig, (a1, a2) = P.plt.subplots(2, 1, figsize=(P.WIDTH, 2.95), sharex=True, gridspec_kw={"hspace": 0.18})
    for ax, key, label in ((a1, "recall", "Recall (%)"), (a2, "fpr", "FPR (%)")):
        v = [100 * r[key] for r in grid]
        ax.vlines(ts, [100 * r[f"{key}_lo"] for r in grid], [100 * r[f"{key}_hi"] for r in grid], color=P.NAVY, lw=0.9, zorder=2)
        ax.plot(ts, v, "-o", color=P.NAVY, lw=1.4, ms=3.8, mec="white", mew=0.8, zorder=3)
        ax.axvline(current, color=P.INK2, lw=0.9, ls="--", zorder=1)
        if tau_star is not None:
            r = next(r for r in grid if r["threshold"] == tau_star)
            ax.plot([tau_star], [100 * r[key]], "o", ms=8.5, mfc="none", mec=P.ORANGE, mew=1.5, zorder=4)
        ax.set_ylabel(label)
        P.tidy(ax)
    lo_all = min(100 * r["recall_lo"] for r in grid)
    a1.set_ylim(max(0.0, lo_all - (0.6 if lo_all > 95 else 3)), 100.4)
    a2.axhline(100 * fpr_limit, color=P.ORANGE, lw=0.9, ls="--", zorder=1)
    a2.text(ts[-1], 100 * fpr_limit, "FPR limit 2% ", color=P.INK2, va="bottom", ha="right", fontsize=6.4)
    a2.set_ylim(0, max(100 * fpr_limit * 1.35, 1.15 * max(100 * r["fpr_hi"] for r in grid)))
    a1.text(current, a1.get_ylim()[1], f" current {current:g}", color=P.INK2, va="top", ha="left", fontsize=6.4)
    if tau_star is not None:
        a2.text(tau_star, a2.get_ylim()[1] * 0.97, f" τ* = {tau_star:g} ", color=P.INK, va="top",
                ha="left" if tau_star < ts[-2] else "right", fontsize=6.4)
    a2.set_xticks(ts, [f"{t:g}" for t in ts])
    a2.set_xlabel("Threshold τ (validation set; whiskers: 95% Wilson intervals)")
    return P.save(fig, folder, "fig_threshold_sensitivity")


def main(argv=None) -> int:
    args = parse_args(argv)
    root = C.find_project_root(args.project_root)
    out = C.out_dir(root, args.out_dir)
    data_dir = Path(args.data_dir) if args.data_dir else C.default_data_dir(root)
    model_dir = C.find_checkpoint(root, args.detector_dir)
    thresholds = sorted(set(round(t, 6) for t in args.thresholds) | {round(args.current, 6)})

    # saved test predictions (final evaluation only) and the data
    pred_paths = {"regular": Path(args.regular_predictions) if args.regular_predictions else root / "results" / "layer1_predictions.csv",
                  "challenge": Path(args.challenge_predictions) if args.challenge_predictions else root / "results" / "layer1_challenge_predictions.csv"}
    tests, test_cols = {}, {}
    for split, path in pred_paths.items():
        if not path.exists():
            raise SystemExit(f"Saved predictions not found: {path}. They are written by train_layer1.py; "
                             f"pass --{split}-predictions if they are elsewhere.")
        tests[split], test_cols[split] = C.load_saved_predictions(path, args.label_column, args.score_column, args.family_column)
    print("Benchmark partitions:")
    parts = C.load_partitions(data_dir, ("validation", "regular"),
                              {"validation": args.validation_csv, "regular": args.regular_csv})

    # 1. detector, checked against the saved regular-test predictions
    detector, scorer = C.aligned_detector(root, out, model_dir, tests["regular"], parts["regular"], args.batch_size)

    # 2. validation scores (the only data that influences the choice)
    rows = parts["validation"]
    cols = list(rows[0].keys())
    tc = C.pick_column(cols, C.TEXT_CANDIDATES, what="text", exclude=r"clean|reference|payload")
    lc = C.pick_column(cols, C.LABEL_CANDIDATES, args.label_column, what="label", exclude=r"pred")
    fc = C.pick_column(cols, C.FAMILY_CANDIDATES, args.family_column, what="family", required=False)
    ic = C.pick_column(cols, C.ID_CANDIDATES, required=False)
    cache, meta_file = out / "validation_scores.csv", out / "validation_scores.meta.json"
    key = {"checkpoint": scorer["signature"], "windows": [detector.max_length, detector.stride],
           "rows": len(rows), "first": rows[0].get(ic, ""), "last": rows[-1].get(ic, "")}
    if cache.exists() and meta_file.exists() and json.loads(meta_file.read_text()).get("key") == key:
        val = [{"chunk_id": r["chunk_id"], "label": int(r["label"]), "family": r["family"], "score": float(r["score"])}
               for r in C.read_rows(cache)]
        print(f"Validation scores: reusing {cache.name} (same checkpoint, windows and data)")
    else:
        print(f"Scoring {len(rows):,} validation chunks ...")
        start = time.perf_counter()
        scores = detector.score([r[tc] for r in rows], "validation chunks")
        val = [{"chunk_id": r.get(ic, str(i)) if ic else str(i), "label": C.to_label(r[lc]), "family": C.family_of(r, fc),
                "score": s} for i, (r, s) in enumerate(zip(rows, scores))]
        C.write_rows(cache, val, ["chunk_id", "label", "family", "score"])
        meta_file.write_text(json.dumps({"key": key, "seconds": round(time.perf_counter() - start, 1)}, indent=2))

    # 3. sweep and 4. selection
    labels, scores = [r["label"] for r in val], [r["score"] for r in val]
    grid = []
    for t in thresholds:
        r = C.rates(C.confusion(labels, scores, t))
        r["threshold"] = t
        grid.append(r)
    tau_star, feasible = select_threshold(grid, args.fpr_limit)
    for r in grid:
        r["feasible"] = int(r["threshold"] in feasible)
        r["selected"] = int(r["threshold"] == tau_star)
        r["current"] = int(abs(r["threshold"] - args.current) < 1e-9)
    print(f"\nValidation ({len(val):,} chunks: {sum(labels):,} injected, {len(labels) - sum(labels):,} benign)")
    print(f"  {'tau':>6} {'recall':>8} {'FPR':>8}  FPR<=2%")
    for r in grid:
        print(f"  {r['threshold']:>6g} {C.pct(r['recall']):>8} {C.pct(r['fpr']):>8}  {'yes' if r['feasible'] else 'no '}"
              f"{'  <- selected' if r['selected'] else ''}{'  (current)' if r['current'] else ''}")
    print(f"Selected on validation: tau* = {tau_star:g}" if tau_star is not None else
          "No threshold in the grid meets the FPR limit on validation.")

    # 5. final evaluation on the test sets
    eval_thresholds = sorted({args.current} | ({tau_star} if tau_star is not None else set()))
    final, family_rows = [], []
    for split in ("regular", "challenge"):
        y, s = [r["label"] for r in tests[split]], [r["score"] for r in tests[split]]
        for t in eval_thresholds:
            role = ("selected and current" if t == tau_star and abs(t - args.current) < 1e-9
                    else "selected" if t == tau_star else "current")
            final.append({"split": split, "threshold": t, "role": role, **C.rates(C.confusion(y, s, t))})
    fam: dict[str, list[float]] = {}
    for r in tests["challenge"]:
        if r["label"] == 1:
            fam.setdefault(r["family"] or "(no family column)", []).append(r["score"])
    for name in sorted(fam):
        for t in eval_thresholds:
            hit = sum(s >= t for s in fam[name])
            lo, hi = C.wilson(hit, len(fam[name]))
            family_rows.append({"family": name, "threshold": t, "n": len(fam[name]), "detected": hit,
                                "missed": len(fam[name]) - hit, "recall": hit / len(fam[name]), "recall_lo": lo, "recall_hi": hi})
    print("\nFinal evaluation on the test sets (not used for the choice):")
    for r in final:
        print(f"  {r['split']:<9} tau={r['threshold']:<6g} ({r['role']}): recall {C.pct(r['recall'])} "
              f"[{C.pct(r['recall_lo'])}, {C.pct(r['recall_hi'])}], FPR {C.pct(r['fpr'])}, accuracy {C.pct(r['accuracy'])}")

    C.write_rows(out / "threshold_sensitivity_validation.csv", grid,
                 ["threshold", "TP", "FN", "FP", "TN", "positives", "negatives", "recall", "recall_lo", "recall_hi", "fpr",
                  "fpr_lo", "fpr_hi", "precision", "accuracy", "f1", "feasible", "selected", "current"])
    C.write_rows(out / "threshold_final_evaluation.csv", final,
                 ["split", "threshold", "role", "TP", "FN", "FP", "TN", "accuracy", "accuracy_lo", "accuracy_hi", "precision",
                  "recall", "recall_lo", "recall_hi", "fpr", "fpr_lo", "fpr_hi", "f1"])
    C.write_rows(out / "threshold_family_recall.csv", family_rows,
                 ["family", "threshold", "n", "detected", "missed", "recall", "recall_lo", "recall_hi"])
    figure = plot(grid, tau_star, args.current, args.fpr_limit, out / "figures_python")
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "detector_dir": str(model_dir), "detector": detector.description,
        "checkpoint_threshold": detector.saved_threshold, "scorer_check": scorer["verification"],
        "validation": {"chunks": len(val), "injected": sum(labels), "benign": len(labels) - sum(labels),
                       "columns": {"text": tc, "label": lc, "family": fc, "id": ic}},
        "thresholds": thresholds, "fpr_limit": args.fpr_limit, "current_threshold": args.current,
        "rule": "highest validation recall with validation FPR <= limit; ties: lower FPR, then the centre of the tied range",
        "feasible_thresholds": feasible, "selected_threshold": tau_star,
        "selected_equals_current": tau_star is not None and abs(tau_star - args.current) < 1e-9,
        "grid": grid, "final_evaluation": final, "family_recall": family_rows,
        "test_sources": {k: str(v) for k, v in pred_paths.items()}, "test_columns": test_cols, "figure": str(figure),
    }
    (out / "threshold_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote threshold_sensitivity_validation.csv, threshold_final_evaluation.csv, threshold_family_recall.csv,"
          f" threshold_summary.json and figures_python/{figure.name} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
