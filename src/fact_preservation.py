"""Phase 3 — fact-preservation verification for the Layer-2 sanitizer.

For every poisoned test chunk we compare the sanitized output against the chunk's
original benign portion (`clean_fact`):

  cos_clean      cosine(emb(sanitized), emb(clean_fact))        higher = facts kept
  rougeL_recall  ROUGE-L recall of clean_fact in sanitized      higher = facts kept
  rougeL_f1      ROUGE-L F1                                     penalizes added notices
  cos_payload    cosine(emb(sanitized), emb(payload))           lower  = attack removed
  payload_resid  ROUGE-L recall of payload in sanitized          lower  = attack removed
  canary_removed canary string no longer present                 1 = neutralized

We also run the sanitizer on BENIGN test chunks (as if Layer 1 had mis-flagged
them) to measure collateral damage on false positives.

Usage:
  python -m src.fact_preservation --n 300                   # rules + SLM fallback, no detector
  python -m src.fact_preservation --n 300 --with-detector   # full Layer-2 with Layer-1 localization
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from src.metrics import rouge_l
from src.payloads import canary_hit
from src.rag_common import make_embeddings
from src.utils import get_logger, load_config, read_jsonl, set_seed

log = get_logger("fact_preservation")


class FactPreservationScorer:
    def __init__(self, cfg: dict):
        self.emb = make_embeddings(cfg)

    def _cos(self, a: list[str], b: list[str]) -> np.ndarray:
        ea = np.asarray(self.emb.embed_documents(a), dtype=np.float32)
        eb = np.asarray(self.emb.embed_documents(b), dtype=np.float32)
        ea /= np.linalg.norm(ea, axis=1, keepdims=True) + 1e-9
        eb /= np.linalg.norm(eb, axis=1, keepdims=True) + 1e-9
        return (ea * eb).sum(axis=1)

    def score_batch(self, sanitized: list[str], clean: list[str], payloads: list[str], canaries: list[str]) -> list[dict]:
        cos_clean = self._cos(sanitized, clean)
        has_payload = [bool(p) for p in payloads]
        cos_payload = self._cos(sanitized, [p or " " for p in payloads])
        rows = []
        for i, (s, c, p, k) in enumerate(zip(sanitized, clean, payloads, canaries)):
            r = rouge_l(c, s)
            rows.append({
                "cos_clean": float(cos_clean[i]), "rougeL_recall": r["recall"], "rougeL_f1": r["f1"],
                "cos_payload": float(cos_payload[i]) if has_payload[i] else None,
                "payload_resid": rouge_l(p, s)["recall"] if p else None,
                "canary_removed": (not canary_hit(s, k)) if k else None,
            })
        return rows


def _mean(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r[key] is not None]
    return float(mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--with-detector", action="store_true")
    ap.add_argument("--no-slm", action="store_true")
    ap.add_argument("--mode", choices=["strip", "describe"], default=None)
    args = ap.parse_args()

    from src.layer2_quarantine import QuarantineSanitizer

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    if args.mode:
        cfg["layer2"]["mode"] = args.mode
    det = None
    if args.with_detector:
        from src.layer1_detector import Layer1Detector

        det = Layer1Detector.from_config(cfg)
    san = QuarantineSanitizer(cfg, detector=det, load_slm=not args.no_slm)
    scorer = FactPreservationScorer(cfg)

    test = [r for r in read_jsonl(Path(cfg["paths"]["processed_dir"]) / "chunks.jsonl") if r["split"] == "test"]
    poisoned = [r for r in test if r["label"] == 1][: args.n]
    benign = [r for r in test if r["label"] == 0][: args.n]

    report = {}
    for name, rows in (("poisoned", poisoned), ("benign_misflagged", benign)):
        results = [san.sanitize(r["text"]) for r in rows]
        scores = scorer.score_batch([x.sanitized for x in results], [r["clean_fact"] for r in rows],
                                    [r["payload"] for r in rows], [r["canary"] for r in rows])
        methods: dict[str, int] = {}
        for x in results:
            methods[x.method] = methods.get(x.method, 0) + 1
        report[name] = {
            "n": len(rows),
            "cos_clean": _mean(scores, "cos_clean"), "rougeL_recall": _mean(scores, "rougeL_recall"),
            "rougeL_f1": _mean(scores, "rougeL_f1"), "cos_payload": _mean(scores, "cos_payload"),
            "payload_resid": _mean(scores, "payload_resid"), "canary_removed_rate": _mean(scores, "canary_removed"),
            "changed_rate": float(mean(x.changed for x in results)) if results else None,
            "latency_ms_mean": float(mean(x.latency_ms for x in results)) if results else None,
            "methods": methods,
        }
        # Reference point: the poisoned chunk *before* sanitization.
        if name == "poisoned":
            base = scorer.score_batch([r["text"] for r in rows], [r["clean_fact"] for r in rows],
                                      [r["payload"] for r in rows], [r["canary"] for r in rows])
            report["poisoned_unsanitized_reference"] = {k: _mean(base, k) for k in
                                                        ("cos_clean", "rougeL_recall", "payload_resid", "canary_removed")}
        for r, x in list(zip(rows, results))[:3]:
            log.info("[%s] %s\n  BEFORE: %s\n  AFTER : %s", name, x.method, r["text"][:300], x.sanitized[:300])

    out = Path(cfg["paths"]["results_dir"]) / "fact_preservation.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
