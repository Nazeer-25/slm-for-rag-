"""Fine-tune DistilBERT as the Layer-1 detector and evaluate it on the held-out test set.

Reads data/train.csv (140 examples) and data/test.csv (60 examples). Writes
  models/layer1_detector/        model weights, tokenizer, detector_config.json (threshold 0.50,
                                 window length and stride)
  results/detector_metrics.json  accuracy, precision, recall, F1, ROC-AUC, FPR, confusion matrix,
                                 per-family / per-position recall, misclassified examples

Takes about a minute on a normal CPU. The first run downloads distilbert-base-uncased.
Stop the website first if it is running (it holds the old model in memory).

Usage:
  python src/train_layer1.py
  python src/train_layer1.py --epochs 6
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

from config import (DETECTOR_BASE_MODEL, DETECTOR_MAX_LENGTH, DETECTOR_STRIDE, DETECTOR_THRESHOLD, PATHS,
                    Paths)
from data_generator import read_csv
from layer1_detector import CONFIG_NAME, Layer1Detector


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    """Probability that a random positive outscores a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    preds = [int(s >= threshold) for s in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(preds, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": roc_auc(labels, scores),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def train(paths: Paths, base_model: str, epochs: int, lr: float, batch_size: int, seed: int,
          max_length: int) -> list[float]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_csv(paths.train_csv)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label={0: "BENIGN", 1: "INJECTION"}, label2id={"BENIGN": 0, "INJECTION": 1})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    enc = tokenizer([r["text"] for r in rows], truncation=True, max_length=max_length, padding=True,
                    return_tensors="pt")
    labels = torch.tensor([r["label"] for r in rows])
    loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], labels), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    losses = []
    print(f"Training {base_model} on {len(rows)} examples ({device}, {epochs} epochs)")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for input_ids, attention_mask, y in loader:
            out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += out.loss.item() * len(y)
        losses.append(running / len(rows))
        print(f"  epoch {epoch}/{epochs}  loss {losses[-1]:.4f}")

    # Write to a fresh directory, then swap it in.
    staging = paths.detector_dir.with_name(paths.detector_dir.name + "_new")
    if staging.exists():
        shutil.rmtree(staging)
    model.save_pretrained(str(staging))
    tokenizer.save_pretrained(str(staging))
    (staging / CONFIG_NAME).write_text(json.dumps({
        "base_model": base_model, "threshold": DETECTOR_THRESHOLD, "max_length": max_length,
        "stride": DETECTOR_STRIDE, "labels": {"0": "benign data", "1": "imperative injection"},
        "trained_on": "data/train.csv", "train_examples": len(rows), "epochs": epochs, "learning_rate": lr,
        "seed": seed, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    try:
        if paths.detector_dir.exists():
            shutil.rmtree(paths.detector_dir)
        staging.rename(paths.detector_dir)
    except PermissionError:
        raise SystemExit(f"Could not replace {paths.detector_dir}: files are in use. Stop the website "
                         f"(Ctrl+C) and run training again. The new model is in {staging}.")
    return losses


def evaluate(paths: Paths) -> dict:
    """Score data/test.csv with the saved detector, exactly as the website will run it."""
    rows = read_csv(paths.test_csv)
    detector = Layer1Detector(paths.detector_dir)
    start = time.perf_counter()
    results = detector.score([r["text"] for r in rows])
    elapsed = (time.perf_counter() - start) * 1000
    labels = [r["label"] for r in rows]
    scores = [res["probability"] for res in results]
    metrics = binary_metrics(labels, scores, detector.threshold)

    by_family: dict = defaultdict(lambda: [0, 0])
    by_position: dict = defaultdict(lambda: [0, 0])
    for r, res in zip(rows, results):
        if r["label"] == 1:
            for key, table in ((r["template_family"], by_family), (r["injection_position"], by_position)):
                table[key][0] += int(res["flagged"])
                table[key][1] += 1
    misclassified = [{"chunk_id": r["chunk_id"], "label": r["label"], "probability": round(res["probability"], 4)}
                     for r, res in zip(rows, results) if int(res["flagged"]) != r["label"]]
    return {
        "model": detector.name, "threshold": detector.threshold, "test_file": "data/test.csv",
        "test_examples": len(rows), "test_injections": sum(labels), "test_benign": len(labels) - sum(labels),
        **metrics,
        "recall_by_family": {k: {"detected": a, "total": b} for k, (a, b) in sorted(by_family.items())},
        "recall_by_position": {k: {"detected": a, "total": b} for k, (a, b) in sorted(by_position.items())},
        "misclassified": misclassified,
        "inference_ms_per_chunk": round(elapsed / max(1, len(rows)), 2),
        "detector_config": detector.config,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveat": "The test set is synthetic and shares template families with training. A high score here "
                  "is not evidence of robustness against unseen injection styles.",
    }


def main(argv: list[str] | None = None, paths: Paths = PATHS) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=DETECTOR_BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=DETECTOR_MAX_LENGTH)
    args = ap.parse_args(argv)
    if not paths.train_csv.exists() or not paths.test_csv.exists():
        raise SystemExit("data/train.csv or data/test.csv is missing. Run: python src/data_generator.py")

    losses = train(paths, args.base_model, args.epochs, args.lr, args.batch_size, args.seed, args.max_length)
    report = evaluate(paths)
    report["train_loss_per_epoch"] = [round(x, 5) for x in losses]
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.detector_metrics_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cm = report["confusion_matrix"]
    print(f"\nTest set ({report['test_examples']} examples, threshold {report['threshold']:.2f}):")
    print(f"  accuracy {report['accuracy']:.3f}  precision {report['precision']:.3f}  recall {report['recall']:.3f}  "
          f"F1 {report['f1']:.3f}  ROC-AUC {report['roc_auc'] if report['roc_auc'] is None else round(report['roc_auc'], 3)}  "
          f"FPR {report['false_positive_rate']:.3f}")
    print(f"  confusion matrix: TN {cm['tn']}  FP {cm['fp']}  FN {cm['fn']}  TP {cm['tp']}")
    print(f"Model saved to {paths.detector_dir}; metrics in {paths.detector_metrics_json}")
    print("Note: synthetic test data shares template families with training; this is not a robustness claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
