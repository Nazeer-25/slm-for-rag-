"""Phase 4 — comparative evaluation harness.

Runs the same queries through the same retriever/index for every setup:

  undefended  plain RAG (vulnerable baseline)
  delimiter   XML-tagging / spotlighting prompt defense
  llm_judge   every chunk screened by an extra main-LLM call, flagged chunks dropped
  hybrid      ours: Layer 1 SLM detector + conditional Layer 2 quarantine
  l1_drop     ablation: Layer 1 only, flagged chunks dropped
  l2_only     ablation: Layer 2 rules on every chunk, no detector

Outputs (results/):
  <setup>_summary.json, logs/<setup>_records.jsonl, comparison.csv, comparison.md,
  asr_by_family.csv, figures/*.png

Usage:
  python eval/run_benchmarks.py                                   # all config setups
  python eval/run_benchmarks.py --setups undefended delimiter hybrid --limit 50
  python eval/run_benchmarks.py --provider ollama
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.baseline_rag import UndefendedRAG  # noqa: E402
from src.defenses import DelimiterRAG, LLMJudgeRAG  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.metrics import aggregate, evaluate_pipeline  # noqa: E402
from src.rag_common import load_retriever  # noqa: E402
from src.utils import get_logger, load_config, read_jsonl, set_seed, write_jsonl  # noqa: E402

log = get_logger("benchmark")

ALL_SETUPS = ["undefended", "delimiter", "llm_judge", "hybrid", "l1_drop", "l2_only"]
TABLE_COLS = ["setup", "ASR_%", "ASR_given_retrieved_%", "ASR_unseen_payloads_%", "utility_benign_%",
              "utility_under_attack_%", "hijack_rate_benign_queries_%", "chunk_FPR_%", "chunk_TPR_%", "fact_retention_rougeL_recall",
              "payload_residue_rougeL_recall", "defense_ms_mean", "latency_overhead_ms", "total_ms_mean",
              "llm_calls_per_query", "tokens_per_query", "cost_usd_per_1k_queries"]


class _Lazy:
    """Load the Layer-1 model and sanitizer once and share them across setups."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._det = self._san = self._san_nodet = None

    @property
    def detector(self):
        if self._det is None:
            from src.layer1_detector import Layer1Detector

            self._det = Layer1Detector.from_config(self.cfg)
        return self._det

    @property
    def sanitizer(self):
        if self._san is None:
            from src.layer2_quarantine import QuarantineSanitizer

            self._san = QuarantineSanitizer(self.cfg, detector=self.detector)
        return self._san

    @property
    def sanitizer_rules_only(self):
        if self._san_nodet is None:
            from src.layer2_quarantine import QuarantineSanitizer

            self._san_nodet = QuarantineSanitizer(self.cfg, detector=None, load_slm=False)
        return self._san_nodet


def build_setup(name: str, retriever, llm, lazy: _Lazy, judge_llm=None):
    if name == "undefended":
        return UndefendedRAG(retriever, llm)
    if name == "delimiter":
        return DelimiterRAG(retriever, llm)
    if name == "llm_judge":
        return LLMJudgeRAG(retriever, llm, judge_llm=judge_llm)
    from src.defended_rag import HybridDefendedRAG, Layer1DropRAG, Layer2OnlyRAG

    if name == "hybrid":
        return HybridDefendedRAG(retriever, llm, lazy.detector, lazy.sanitizer)
    if name == "l1_drop":
        return Layer1DropRAG(retriever, llm, lazy.detector)
    if name == "l2_only":
        return Layer2OnlyRAG(retriever, llm, lazy.sanitizer_rules_only)
    raise ValueError(f"unknown setup {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--setups", nargs="+", choices=ALL_SETUPS, default=None)
    ap.add_argument("--provider", choices=["openai", "ollama", "mock"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="max attack AND max benign queries")
    ap.add_argument("--no-rebuild", action="store_true", help="reuse persisted vector index")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    setups = args.setups or cfg["eval"]["setups"]
    results_dir = Path(cfg["paths"]["results_dir"])

    queries = read_jsonl(Path(cfg["paths"]["processed_dir"]) / "rag_eval.jsonl")
    if args.limit:
        queries = [q for q in queries if q["is_attack"]][: args.limit] + \
                  [q for q in queries if not q["is_attack"]][: args.limit]
    log.info("%d queries (%d attack / %d benign); setups=%s", len(queries), sum(q["is_attack"] for q in queries),
             sum(not q["is_attack"] for q in queries), setups)

    retriever = load_retriever(cfg, rebuild=not args.no_rebuild)
    llm = LLMClient(cfg, provider=args.provider)
    judge_for_asr = llm if cfg["eval"]["use_llm_judge_for_asr"] else None
    lazy = _Lazy(cfg)

    summaries, family_rows = {}, []
    for name in setups:
        pipeline = build_setup(name, retriever, llm, lazy, judge_llm=llm)
        records = evaluate_pipeline(pipeline, queries, judge_llm=judge_for_asr, desc=name)
        summary = aggregate(records)
        summaries[name] = summary
        write_jsonl(results_dir / "logs" / f"{name}_records.jsonl", records)
        (results_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
        for fam, asr in summary["ASR_by_family_%"].items():
            family_rows.append({"setup": name, "attack_family": fam, "ASR_%": asr})
        log.info("%-10s ASR=%5.1f%%  utility=%5.1f%%  FPR=%5.1f%%  defense=%6.1f ms  $/1k=%.4f", name,
                 summary["ASR_%"] or 0, summary["utility_benign_%"] or 0, summary["chunk_FPR_%"] or 0,
                 summary["defense_ms_mean"], summary["cost_usd_per_1k_queries"])

    base_ms = summaries.get("undefended", {}).get("total_ms_mean")
    rows = []
    for name, s in summaries.items():
        row = {"setup": name, **{k: s.get(k) for k in TABLE_COLS if k in s}}
        row["latency_overhead_ms"] = (s["total_ms_mean"] - base_ms) if base_ms is not None else s["defense_ms_mean"]
        rows.append(row)
    df = pd.DataFrame(rows)[TABLE_COLS]
    df.to_csv(results_dir / "comparison.csv", index=False)
    (results_dir / "comparison.md").write_text(df.round(3).to_markdown(index=False))
    pd.DataFrame(family_rows).to_csv(results_dir / "asr_by_family.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n" + df.round(2).to_string(index=False))
    if "delimiter" in summaries:
        br = summaries["delimiter"]["ASR_by_family_%"].get("delimiter_breakout")
        print(f"\nDelimiter defense vs `delimiter_breakout` payloads: ASR = {br if br is None else round(br, 1)}%")
    if not args.no_plots:
        from eval.plot_results import plot_all

        plot_all(results_dir)
        print(f"Figures written to {results_dir / 'figures'}")


if __name__ == "__main__":
    main()
