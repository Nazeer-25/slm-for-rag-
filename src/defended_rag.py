"""Phase 4 — the Hybrid Dual-Layer defended RAG pipeline.

  query -> retrieve top-k -> Layer 1 (batch SLM detector)
        -> clean chunks pass through unchanged
        -> flagged chunks -> Layer 2 (contextual quarantine; never dropped)
        -> main LLM (OpenAI / Ollama, black box) -> answer

The defense sits strictly between the vector DB and the main LLM and needs no
access to the LLM's weights, logits or hidden states. It adds zero main-LLM calls.

Also provides two ablations used by the benchmark:
  Layer1DropRAG  Layer 1 only, flagged chunks dropped (shows context loss)
  Layer2OnlyRAG  Layer 2 rules on every chunk, no detector (shows FPR/latency cost)

Usage:
  python -m src.defended_rag --ask "What is the minimum password length at Initech?"
"""
from __future__ import annotations

import argparse

from src.defenses import DelimiterRAG
from src.layer1_detector import Layer1Detector
from src.layer2_quarantine import QuarantineSanitizer
from src.rag_common import BaseRAG, ChunkTrace, RAGResult


class HybridDefendedRAG(BaseRAG):
    name = "hybrid"

    def __init__(self, retriever, llm, detector: Layer1Detector, sanitizer: QuarantineSanitizer,
                 prompt_style: str = "plain"):
        super().__init__(retriever, llm)
        self.detector = detector
        self.sanitizer = sanitizer
        self.prompt_style = prompt_style

    def build_messages(self, chunks: list[str], question: str) -> list[dict]:
        if self.prompt_style == "delimiter":
            return DelimiterRAG.build_messages(self, chunks, question)
        return super().build_messages(chunks, question)

    def defend(self, traces: list[ChunkTrace], question: str, result: RAGResult) -> None:
        detections = self.detector.detect([t.original_text for t in traces])
        layer2_ms = 0.0
        methods = []
        for t, d in zip(traces, detections):
            t.confidence = d.confidence_score
            t.flagged = d.is_suspicious
            if not d.is_suspicious:
                t.delivered_text, t.action = t.original_text, "pass"
                continue
            res = self.sanitizer.sanitize(t.original_text)
            layer2_ms += res.latency_ms
            t.delivered_text = res.sanitized
            t.action = "sanitized"
            t.method = res.method
            t.neutralized_spans = [(v.start, v.end, v.reasons) for v in res.neutralized]
            methods.append(res.method)
        result.extra.update({
            "layer1_ms": sum(d.latency_ms for d in detections), "layer2_ms": layer2_ms,
            "layer1_scores": [round(d.confidence_score, 4) for d in detections], "layer2_methods": methods,
        })


class Layer1DropRAG(BaseRAG):
    name = "l1_drop"

    def __init__(self, retriever, llm, detector: Layer1Detector):
        super().__init__(retriever, llm)
        self.detector = detector

    def defend(self, traces, question, result) -> None:
        for t, d in zip(traces, self.detector.detect([t.original_text for t in traces])):
            t.flagged, t.confidence = d.is_suspicious, d.confidence_score
            t.action = "dropped" if d.is_suspicious else "pass"
            t.delivered_text = None if d.is_suspicious else t.original_text


class Layer2OnlyRAG(BaseRAG):
    name = "l2_only"

    def __init__(self, retriever, llm, sanitizer: QuarantineSanitizer):
        super().__init__(retriever, llm)
        self.sanitizer = sanitizer

    def defend(self, traces, question, result) -> None:
        for t in traces:
            # Rules only: a sanitizer without a detector never triggers the SLM on
            # chunks where the rules find nothing (see `need_fallback`).
            verdicts = self.sanitizer.analyze(t.original_text)
            if any(v.flagged for v in verdicts):
                t.delivered_text = self.sanitizer.sanitize(t.original_text).sanitized
                t.flagged, t.action = True, "sanitized"
            else:
                t.delivered_text, t.action = t.original_text, "pass"


def build_hybrid(cfg: dict, retriever, llm, prompt_style: str = "plain") -> HybridDefendedRAG:
    detector = Layer1Detector.from_config(cfg)
    sanitizer = QuarantineSanitizer(cfg, detector=detector)
    return HybridDefendedRAG(retriever, llm, detector, sanitizer, prompt_style=prompt_style)


def main() -> None:
    from src.llm_client import LLMClient
    from src.rag_common import load_retriever
    from src.utils import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--provider", choices=["openai", "ollama", "mock"], default=None)
    ap.add_argument("--ask", required=True)
    ap.add_argument("--no-rebuild", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rag = build_hybrid(cfg, load_retriever(cfg, rebuild=not args.no_rebuild), LLMClient(cfg, provider=args.provider))
    res = rag.answer(args.ask)
    for c in res.chunks:
        print(f"--- {c.chunk_id} poisoned={bool(c.label)} p_inj={c.confidence:.3f} action={c.action}")
        if c.action == "sanitized":
            print(f"    BEFORE: {c.original_text[:400]}\n    AFTER : {c.delivered_text[:400]}")
    print(f"\nANSWER: {res.answer}")
    print(f"latency: retrieval={res.retrieval_ms:.0f} ms  defense={res.defense_ms:.0f} ms "
          f"(L1={res.extra['layer1_ms']:.0f}, L2={res.extra['layer2_ms']:.0f})  generation={res.generation_ms:.0f} ms")


if __name__ == "__main__":
    main()
