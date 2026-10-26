"""Phase 1 — undefended (vulnerable) baseline RAG + attack reproduction.

LangChain + ChromaDB/FAISS + all-MiniLM-L6-v2, generation via OpenAI gpt-4o-mini
or a local Ollama model. Retrieved chunks are pasted verbatim into the prompt,
which is exactly what indirect prompt injection exploits.

Usage:
  python -m src.baseline_rag --ask "How many days do customers have to request a refund from Acme Corp?"
  python -m src.baseline_rag --eval                 # baseline ASR on data/processed/rag_eval.jsonl
  python -m src.baseline_rag --eval --provider ollama --limit 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.llm_client import LLMClient
from src.metrics import aggregate, evaluate_pipeline
from src.rag_common import BaseRAG, load_retriever
from src.utils import get_logger, load_config, read_jsonl, set_seed, write_jsonl

log = get_logger("baseline_rag")


class UndefendedRAG(BaseRAG):
    """Retrieve top-k, concatenate, generate. No defense whatsoever."""

    name = "undefended"


def evaluate_attack_success(cfg: dict, pipeline: BaseRAG, limit: int | None = None, judge_llm=None) -> dict:
    """Run every query in rag_eval.jsonl through `pipeline` and compute the baseline ASR."""
    queries = read_jsonl(Path(cfg["paths"]["processed_dir"]) / "rag_eval.jsonl")
    if limit:
        atk = [q for q in queries if q["is_attack"]][:limit]
        ben = [q for q in queries if not q["is_attack"]][:limit]
        queries = atk + ben
    records = evaluate_pipeline(pipeline, queries, judge_llm=judge_llm)
    summary = aggregate(records)
    out_dir = Path(cfg["paths"]["results_dir"])
    write_jsonl(out_dir / "logs" / f"{pipeline.name}_records.jsonl", records)
    (out_dir / f"{pipeline.name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--provider", choices=["openai", "ollama", "mock"], default=None)
    ap.add_argument("--ask", help="answer a single question")
    ap.add_argument("--eval", action="store_true", help="compute baseline ASR")
    ap.add_argument("--limit", type=int, default=None, help="max attack AND max benign queries")
    ap.add_argument("--no-rebuild", action="store_true", help="reuse the persisted vector index")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    llm = LLMClient(cfg, provider=args.provider)
    rag = UndefendedRAG(load_retriever(cfg, rebuild=not args.no_rebuild), llm)

    if args.ask:
        res = rag.answer(args.ask)
        for c in res.chunks:
            print(f"--- {c.chunk_id} (poisoned={bool(c.label)})\n{c.original_text}\n")
        print(f"ANSWER: {res.answer}\n[{res.total_ms:.0f} ms, {res.prompt_tokens}+{res.completion_tokens} tokens]")
    if args.eval:
        judge = llm if cfg["eval"]["use_llm_judge_for_asr"] else None
        summary = evaluate_attack_success(cfg, rag, args.limit, judge_llm=judge)
        print(json.dumps(summary, indent=2))
        print(f"\nBASELINE ASR = {summary['ASR_%']:.1f}%  (given poison retrieved: "
              f"{summary['ASR_given_retrieved_%'] or 0:.1f}%)  | benign utility = {summary['utility_benign_%'] or 0:.1f}%")
    if not args.ask and not args.eval:
        ap.print_help()


if __name__ == "__main__":
    main()
