"""Phase 2 (alternative) — QLoRA fine-tune of Llama-3-8B as the Layer-1 classifier.

Fits a free Colab/Kaggle T4 (16 GB): 4-bit NF4 base (~5.7 GB), fp16 compute (T4 has
no bf16), LoRA r=16 on all attention + MLP projections, gradient checkpointing,
paged 8-bit AdamW. Expect ~1.5-2 h for `max_train_samples=6000`, 1 epoch.

Prerequisites:
  1. Accept the Llama-3 licence at https://huggingface.co/meta-llama/Meta-Llama-3-8B
  2. `hf auth login` (or HF_TOKEN in .env / Colab secrets)
  3. pip install bitsandbytes   (CUDA only — not in the CPU requirements)

Usage:
  python -m src.train_layer1_qlora
  python -m src.train_layer1_qlora --base-model Qwen/Qwen2.5-1.5B   # smaller, ungated alternative
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.train_layer1 import (binary_metrics, breakdown, build_weighted_trainer_cls, class_weights, load_splits,
                              make_compute_metrics, predict_probs, tokenize_rows, tune_threshold)
from src.utils import get_logger, load_config, set_seed

log = get_logger("train_layer1_qlora")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA needs a CUDA GPU (Colab: Runtime > Change runtime type > T4 GPU).")

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorWithPadding, TrainingArguments)

    cfg = load_config(args.config)
    qcfg = cfg["layer1_qlora"]
    set_seed(cfg["seed"])
    base = args.base_model or qcfg["base_model"]
    out_dir = Path(args.output_dir or qcfg["output_dir"])

    splits = load_splits(cfg, qcfg["max_length"] * 4, sentence_aug=True, seed=cfg["seed"])
    splits["train"] = splits["train"][: qcfg["max_train_samples"]]
    splits["val"] = splits["val"][:2000]
    for s, rows in splits.items():
        c = Counter(r["label"] for r in rows)
        log.info("%-5s n=%6d  benign=%6d  injection=%6d", s, len(rows), c[0], c[1])

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16,
        id2label={0: "BENIGN", 1: "INJECTION"}, label2id={"BENIGN": 0, "INJECTION": 1})
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=qcfg["lora_r"], lora_alpha=qcfg["lora_alpha"],
                      lora_dropout=qcfg["lora_dropout"], bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)  # SEQ_CLS task keeps the `score` head trainable
    model.print_trainable_parameters()

    ds = {s: tokenize_rows(rows, tokenizer, qcfg["max_length"]) for s, rows in splits.items()}
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=qcfg["epochs"], learning_rate=qcfg["lr"],
        per_device_train_batch_size=qcfg["train_batch_size"], per_device_eval_batch_size=qcfg["train_batch_size"] * 2,
        gradient_accumulation_steps=qcfg["grad_accum"], warmup_steps=0.03, lr_scheduler_type="cosine",
        optim="paged_adamw_8bit", fp16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1", logging_steps=10, report_to="none",
        seed=cfg["seed"],
    )
    trainer = build_weighted_trainer_cls()(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"], processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=make_compute_metrics(),
        class_weights=class_weights([r["label"] for r in splits["train"]]),
    )
    trainer.train()

    y_val = np.array([r["label"] for r in splits["val"]])
    p_val = predict_probs(trainer, ds["val"])
    thr = tune_threshold(y_val, p_val, cfg["layer1"]["target_fpr"])
    y_test = np.array([r["label"] for r in splits["test"]])
    p_test = predict_probs(trainer, ds["test"])
    report = {"base_model": base, "val@tuned": binary_metrics(y_val, p_val, thr),
              "test@0.5": binary_metrics(y_test, p_test, 0.5), "test@tuned": binary_metrics(y_test, p_test, thr),
              "test_breakdown@tuned": breakdown(splits["test"], p_test, thr)}

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_dir))  # LoRA adapter + score head only (~100 MB)
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "threshold.json").write_text(json.dumps({"threshold": thr}, indent=2))
    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "test_breakdown@tuned"}, indent=2, default=float))


if __name__ == "__main__":
    main()
