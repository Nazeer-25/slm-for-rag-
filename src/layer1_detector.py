"""Layer 1 - post-retrieval anomaly detector (fine-tuned DistilBERT, 0 = benign data, 1 = injection).

Each retrieved chunk is tokenized into overlapping windows (max_length tokens, `stride` tokens of
overlap), every window is scored, and the chunk's injection probability is the maximum over its
windows, so a payload at the end of a long chunk is never truncated away. Chunks with
probability >= 0.50 are routed to Layer 2; the rest bypass it unchanged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from config import DETECTOR_MAX_LENGTH, DETECTOR_STRIDE, DETECTOR_THRESHOLD, PATHS

CONFIG_NAME = "detector_config.json"


def detector_available(model_dir: Path = PATHS.detector_dir) -> bool:
    return (model_dir / "config.json").is_file() and (model_dir / CONFIG_NAME).is_file()


class Layer1Detector:
    def __init__(self, model_dir: Path = PATHS.detector_dir, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        model_dir = Path(model_dir)
        if not detector_available(model_dir):
            raise FileNotFoundError(f"No trained detector in {model_dir}. Run: python src/train_layer1.py")
        cfg = json.loads((model_dir / CONFIG_NAME).read_text(encoding="utf-8"))
        self.config = cfg
        self.threshold = float(cfg.get("threshold", DETECTOR_THRESHOLD))
        self.max_length = int(cfg.get("max_length", DETECTOR_MAX_LENGTH))
        self.stride = int(cfg.get("stride", DETECTOR_STRIDE))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        self.name = f"{cfg.get('base_model', 'distilbert-base-uncased')} (fine-tuned)"
        self._torch = torch

    def score(self, texts: list[str]) -> list[dict]:
        """One batched pass over all windows of all texts."""
        if not texts:
            return []
        torch = self._torch
        start = time.perf_counter()
        enc = self.tokenizer(list(texts), truncation=True, max_length=self.max_length, stride=self.stride,
                             return_overflowing_tokens=True, padding=True, return_tensors="pt")
        owner = enc.pop("overflow_to_sample_mapping").tolist()
        inputs = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            probs = torch.softmax(self.model(**inputs).logits.float(), dim=-1)[:, 1].cpu().tolist()
        per_text: list[list[float]] = [[] for _ in texts]
        for p, i in zip(probs, owner):
            per_text[i].append(round(float(p), 6))
        latency = (time.perf_counter() - start) * 1000
        return [{"probability": max(w), "flagged": max(w) >= self.threshold, "window_probabilities": w,
                 "windows": len(w), "batch_latency_ms": round(latency, 2)} for w in per_text]
