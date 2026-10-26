"""Phase 2 — fine-tune the Layer-1 SLM anomaly detector (DistilBERT, binary).

  0 = benign data, 1 = imperative injection

Design choices worth reporting:
  * Splits come from data_pipeline (document-level, payload-template-level); the test
    split contains payload families/templates never seen in training.
  * Class-weighted cross-entropy (weights = N / (2 * n_c)).
  * Positives longer than max_length are cropped *around the payload*, so truncation
    never produces a "positive" window that contains no attack.
  * Sentence-level augmentation (payload alone = 1, single benign sentence = 0) lets
    Layer 2 reuse this model to localize the malicious sentence inside a chunk.
  * The decision threshold is tuned on validation to the configured target FPR and
    saved next to the model (threshold.json); Layer 1 loads it automatically.

Usage (Colab T4: ~10-15 min for 3 epochs on the default data):
  python -m src.train_layer1
  python -m src.train_layer1 --epochs 1 --max-train-samples 2000      # quick run
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score,
                             roc_curve)

from src.utils import get_logger, load_config, read_jsonl, resolve_device, set_seed

log = get_logger("train_layer1")


# ----------------------------------------------------------------------------- data

def crop_around_payload(text: str, payload: str, max_chars: int, rng: random.Random) -> str:
    if len(text) <= max_chars:
        return text
    idx = text.find(payload[:60]) if payload else -1
    if idx < 0:
        return text[:max_chars]
    slack = max(0, max_chars - min(len(payload), max_chars))
    start = max(0, min(idx - rng.randint(0, slack), len(text) - max_chars))
    return text[start:start + max_chars]


def load_splits(cfg: dict, max_chars: int, sentence_aug: bool, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    rows = read_jsonl(Path(cfg["paths"]["processed_dir"]) / "chunks.jsonl")
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        text = crop_around_payload(r["text"], r["payload"], max_chars, rng) if r["label"] else r["text"][:max_chars * 2]
        splits[r["split"]].append({"text": text, "label": int(r["label"]), "attack_family": r["attack_family"],
                                   "position": r["position"], "payload_seen": r["payload_seen"],
                                   "source": r["source"]})
    if sentence_aug:
        for split in ("train", "val"):
            src = [r for r in rows if r["split"] == split]
            aug = []
            for r in src:
                if r["label"] and r["payload"]:
                    aug.append({"text": r["payload"], "label": 1, "attack_family": r["attack_family"],
                                "position": "sentence", "payload_seen": True, "source": "aug"})
                elif not r["label"]:
                    sents = [s for s in re.split(r"(?<=[.!?])\s+", r["text"]) if len(s) > 20]
                    if sents:
                        aug.append({"text": rng.choice(sents), "label": 0, "attack_family": "", "position": "",
                                    "payload_seen": False, "source": "aug"})
            splits[split] += aug
    for split in splits:
        rng.shuffle(splits[split])
    return splits


def class_weights(labels: list[int]) -> torch.Tensor:
    counts = Counter(labels)
    n = len(labels)
    return torch.tensor([n / (2 * max(1, counts[c])) for c in (0, 1)], dtype=torch.float)


# ----------------------------------------------------------------------------- metrics

def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = float("nan")
    return {"threshold": float(threshold), "accuracy": accuracy_score(y_true, pred), "precision": p, "recall": r,
            "f1": f1, "roc_auc": auc, "fpr": fp / max(1, fp + tn), "tp": int(tp), "fp": int(fp), "tn": int(tn),
            "fn": int(fn)}


def tune_threshold(y_true: np.ndarray, prob: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose validation FPR <= target (maximizes recall under the FPR budget)."""
    fpr, tpr, thr = roc_curve(y_true, prob)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.5
    best = ok[np.argmax(tpr[ok])]
    # No validation score lies strictly between thr[best] and the next ROC point,
    # so their midpoint has the same val FPR/TPR but a wider margin on unseen data.
    t = thr[best] if best + 1 >= len(thr) else (thr[best] + thr[best + 1]) / 2
    return float(min(max(t, 1e-4), 0.9999))


def softmax_pos(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e[:, 1] / e.sum(axis=1)


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        m = binary_metrics(np.asarray(labels), softmax_pos(np.asarray(logits, dtype=np.float32)))
        return {k: v for k, v in m.items() if k in ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr")}
    return compute_metrics


def breakdown(rows: list[dict], prob: np.ndarray, threshold: float) -> dict:
    """Recall per attack family / position / seen-vs-unseen and FPR per benign source."""
    out: dict[str, dict] = {"recall_by_family": {}, "recall_by_position": {}, "recall_seen_vs_unseen": {},
                            "fpr_by_source": {}}
    groups: dict[tuple[str, str], list[int]] = {}
    for r, p in zip(rows, prob):
        hit = int(p >= threshold)
        if r["label"]:
            groups.setdefault(("recall_by_family", r["attack_family"]), []).append(hit)
            groups.setdefault(("recall_by_position", r["position"]), []).append(hit)
            groups.setdefault(("recall_seen_vs_unseen", "seen" if r["payload_seen"] else "unseen"), []).append(hit)
        else:
            groups.setdefault(("fpr_by_source", r["source"]), []).append(hit)
    for (kind, key), hits in sorted(groups.items()):
        out[kind][key] = {"rate": float(np.mean(hits)), "n": len(hits)}
    return out


# ----------------------------------------------------------------------------- trainer

def build_weighted_trainer_cls():
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.float()
            weight = self._class_weights.to(logits.device) if self._class_weights is not None else None
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 2), labels.view(-1), weight=weight)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def tokenize_rows(rows: list[dict], tokenizer, max_length: int):
    from datasets import Dataset

    ds = Dataset.from_dict({"text": [r["text"] for r in rows], "label": [r["label"] for r in rows]})
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length), batched=True,
                  remove_columns=["text"]).rename_column("label", "labels")


def predict_probs(trainer, ds) -> np.ndarray:
    logits = trainer.predict(ds).predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return softmax_pos(np.asarray(logits, dtype=np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-model", default=None, help="HF id or local path (default: config layer1.base_model)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--no-sentence-aug", action="store_true")
    args = ap.parse_args()

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, TrainingArguments)

    cfg = load_config(args.config)
    lcfg = cfg["layer1"]
    set_seed(cfg["seed"])
    base = args.base_model or lcfg["base_model"]
    out_dir = Path(args.output_dir or lcfg["output_dir"])
    device = resolve_device(lcfg["device"])
    max_chars = lcfg["max_length"] * 4  # ~4 chars/token for English WordPiece

    splits = load_splits(cfg, max_chars, sentence_aug=not args.no_sentence_aug, seed=cfg["seed"])
    if args.max_train_samples:
        splits["train"] = splits["train"][: args.max_train_samples]
    for s, rows in splits.items():
        c = Counter(r["label"] for r in rows)
        log.info("%-5s n=%6d  benign=%6d  injection=%6d", s, len(rows), c[0], c[1])

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, id2label={0: "BENIGN", 1: "INJECTION"}, label2id={"BENIGN": 0, "INJECTION": 1})
    ds = {s: tokenize_rows(rows, tokenizer, lcfg["max_length"]) for s, rows in splits.items()}
    weights = class_weights([r["label"] for r in splits["train"]])
    log.info("class weights: %s | device=%s", weights.tolist(), device)

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs or lcfg["epochs"],
        learning_rate=lcfg["lr"],
        per_device_train_batch_size=lcfg["train_batch_size"],
        per_device_eval_batch_size=lcfg["eval_batch_size"],
        weight_decay=lcfg["weight_decay"],
        warmup_steps=lcfg["warmup_ratio"],  # transformers>=5: float in [0,1) = ratio
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=device == "cuda",
        logging_steps=50,
        report_to="none",
        seed=cfg["seed"],
        use_cpu=device == "cpu",
        dataloader_num_workers=2 if device == "cuda" else 0,
    )
    trainer = build_weighted_trainer_cls()(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_compute_metrics(), class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60

    # Threshold tuning on validation, final report on test.
    y_val = np.array([r["label"] for r in splits["val"]])
    p_val = predict_probs(trainer, ds["val"])
    thr = tune_threshold(y_val, p_val, lcfg["target_fpr"])
    y_test = np.array([r["label"] for r in splits["test"]])
    p_test = predict_probs(trainer, ds["test"])
    report = {
        "base_model": base, "train_minutes": train_min, "device": device,
        "n": {s: len(r) for s, r in splits.items()},
        "val@0.5": binary_metrics(y_val, p_val, 0.5),
        "val@tuned": binary_metrics(y_val, p_val, thr),
        "test@0.5": binary_metrics(y_test, p_test, 0.5),
        "test@tuned": binary_metrics(y_test, p_test, thr),
        "test_breakdown@tuned": breakdown(splits["test"], p_test, thr),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "threshold.json").write_text(json.dumps({"threshold": thr, "target_fpr": lcfg["target_fpr"]}, indent=2))
    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "test_breakdown@tuned"}, indent=2, default=float))
    print(f"\nTuned threshold={thr:.4f} (val FPR <= {lcfg['target_fpr']:.1%}). Model saved to {out_dir}")


if __name__ == "__main__":
    main()
