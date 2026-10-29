"""Single source of truth for every number quoted in the paper.

All inputs are the recorded results in DATASET.md (run of 29 Oct 2026).
Derived values (intervals, MCC, per-family totals, parameter counts) are
computed here so the figures, tables and MATLAB scripts cannot disagree.
"""
import json
import math
from pathlib import Path

Z = 1.959963984540054  # two-sided 95 %


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def metrics(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    acc = (tp + tn) / n
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    spec = tn / (tn + fp)
    f1 = 2 * prec * rec / (prec + rec)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "counts": {"TN": tn, "FP": fp, "FN": fn, "TP": tp, "N": n},
        "accuracy": acc, "accuracy_ci": wilson(tp + tn, n),
        "precision": prec, "precision_ci": wilson(tp, tp + fp),
        "recall": rec, "recall_ci": wilson(tp, tp + fn),
        "specificity": spec, "specificity_ci": wilson(tn, tn + fp),
        "fpr": fp / (tn + fp), "fpr_ci": tuple(1 - x for x in reversed(wilson(tn, tn + fp))),
        "f1": f1, "mcc": mcc,
    }


# ---- DistilBERT-base (uncased, 2 labels) parameter count, exact ----------
V, P, D, H, L = 30522, 512, 768, 3072, 6
emb = V * D + P * D + 2 * D                      # word + position + LayerNorm
block = 4 * (D * D + D) + 2 * D + (D * H + H) + (H * D + D) + 2 * D
head = (D * D + D) + (D * 2 + 2)                 # pre_classifier + classifier
distil_total = emb + L * block + head
distil_trainable = block + head                  # last block + head

# ---- all-MiniLM-L6-v2 (BERT, 6 x 384, with pooler) -------------------------
v, p, d, h, l = 30522, 512, 384, 1536, 6
mini = v * d + p * d + 2 * d + 2 * d + l * (4 * (d * d + d) + 2 * d + (d * h + h) + (h * d + d) + 2 * d) + (d * d + d)

regular = metrics(tn=5000, fp=0, fn=0, tp=5000)
challenge = metrics(tn=5000, fp=0, fn=191, tp=4809)

# Per-family misses on the challenge set (DATASET.md). Totals for the two
# families below are the only integers consistent with the reported recall.
fam_reported = {"Multilingual instructions": (148, 0.7636), "Dependency traps": (37, 0.9391),
                "Unicode spoofing": (6, 0.9905)}
families = {}
for name, (miss, rec) in fam_reported.items():
    ns = [n for n in range(miss + 1, 5001) if round(100 * (1 - miss / n), 2) == round(100 * rec, 2)]
    families[name] = {"missed": miss, "recall": rec, "n_consistent": [min(ns), max(ns)]}
for name in ["Indirect completion", "Reasoning lures", "Polite requests", "Few-shot imitation",
             "Encoded instructions"]:
    families[name] = {"missed": 0, "recall": 1.0, "n_consistent": None}
for name in ("Multilingual instructions", "Dependency traps"):
    lo, hi = families[name]["n_consistent"]
    assert lo == hi, name
    n = lo
    families[name]["n"] = n
    families[name]["recall_ci"] = wilson(n - families[name]["missed"], n)

ml, dep = families["Multilingual instructions"], families["Dependency traps"]
rec_wo_ml = (4809 - (ml["n"] - 148)) / (5000 - ml["n"])

out = {
    "params": {
        "distilbert_total": distil_total, "distilbert_trainable": distil_trainable,
        "trainable_share": distil_trainable / distil_total,
        "distilbert_block": block, "distilbert_head": head, "minilm_total": mini,
        "llama32_3b_total_reported": 3.21e9,
    },
    "training": {"pool": 70000, "sample": 6000, "sample_share": 6000 / 70000, "batch": 16,
                 "epochs_run": 3, "selected_epoch": 1, "threshold": 0.175,
                 "steps_per_epoch": math.ceil(6000 / 16)},
    "regular": regular, "challenge": challenge, "families": families,
    "miss_share": {k: v["missed"] / 191 for k, v in families.items() if v["missed"]},
    "recall_without_multilingual": rec_wo_ml,
    "recall_without_multilingual_and_dependency":
        (4809 - (ml["n"] - 148) - (dep["n"] - 37)) / (5000 - ml["n"] - dep["n"]),
}
path = Path(__file__).with_name("numbers.json")
path.write_text(json.dumps(out, indent=2))


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}%"


print("DistilBERT total", f"{distil_total:,}", "trainable", f"{distil_trainable:,}",
      pct(distil_trainable / distil_total), "| block", f"{block:,}", "head", f"{head:,}")
print("MiniLM total", f"{mini:,}")
for name, m in (("regular", regular), ("challenge", challenge)):
    print(f"[{name}] acc {pct(m['accuracy'])} {tuple(pct(x) for x in m['accuracy_ci'])}"
          f" | prec {pct(m['precision'])} {tuple(pct(x) for x in m['precision_ci'])}"
          f" | rec {pct(m['recall'])} {tuple(pct(x) for x in m['recall_ci'])}"
          f" | FPR {pct(m['fpr'])} {tuple(pct(x, 3) for x in m['fpr_ci'])}"
          f" | F1 {pct(m['f1'])} | MCC {m['mcc']:.4f}")
for k, v in families.items():
    print(k, v)
print("miss share", {k: pct(v, 1) for k, v in out["miss_share"].items()})
print("recall w/o multilingual", pct(rec_wo_ml), "| w/o ML+dep", pct(out["recall_without_multilingual_and_dependency"]))
