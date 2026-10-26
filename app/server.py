"""Web front end for the hybrid IPI defense: FastAPI backend + static single-page UI.

  python -m app.server                                   # uses configs/config.yaml
  python -m app.server --provider ollama --port 8000
  python -m app.server --config configs/smoke.yaml       # fully offline demo (mock LLM)

Then open http://localhost:8000. Endpoints (JSON):
  GET  /api/status       what is loaded (LLM, index, Layer 1, Layer 2) and why not
  GET  /api/examples     sample attack / benign queries from rag_eval.jsonl
  POST /api/query        run one question through several setups side by side
  POST /api/sanitize     Layer-2 playground: sentence verdicts + sanitized text
  GET  /api/results      benchmark tables written by eval/run_benchmarks.py
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.metrics import answer_correct
from src.payloads import canary_hit
from src.utils import get_logger, load_config, read_jsonl

log = get_logger("app")
STATIC = Path(__file__).resolve().parent / "static"
SETUPS = ["undefended", "delimiter", "llm_judge", "hybrid"]


class Engine:
    """Loads every component once; each part fails independently with a readable reason."""

    def __init__(self, cfg: dict, provider: str | None, rebuild_index: bool):
        self.cfg = cfg
        self.lock = threading.Lock()  # spaCy/torch models are shared across requests
        self.status: dict[str, dict] = {}
        self.llm = self.retriever = self.detector = self.sanitizer = self.rules_sanitizer = None
        self._try("llm", lambda: self._load_llm(provider))
        self._try("index", lambda: self._load_index(rebuild_index))
        self._try("layer1", self._load_detector)
        self._try("layer2", self._load_sanitizer)
        self.queries = self._load_queries()

    def _try(self, name: str, fn) -> None:
        try:
            detail = fn()
            self.status[name] = {"ok": True, "detail": detail}
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI instead of crashing
            log.warning("%s unavailable: %s", name, exc)
            self.status[name] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}

    def _load_llm(self, provider):
        from src.llm_client import LLMClient

        self.llm = LLMClient(self.cfg, provider=provider)
        return f"{self.llm.provider} · {self.llm.model}"

    def _load_index(self, rebuild: bool):
        from src.rag_common import load_retriever

        persisted = Path(self.cfg["paths"]["chroma_dir"]) / self.cfg["vector_store"]["backend"]
        self.retriever = load_retriever(self.cfg, rebuild=rebuild or not persisted.exists())
        return f"{self.cfg['vector_store']['backend']} · top-{self.retriever.top_k} · {self.cfg['embeddings']['model_name']}"

    def _load_detector(self):
        from src.layer1_detector import Layer1Detector

        self.detector = Layer1Detector.from_config(self.cfg)
        return f"{Path(self.cfg['layer1']['output_dir']).name} · threshold {self.detector.threshold:.3f} · {self.detector.device}"

    def _load_sanitizer(self):
        from src.layer2_quarantine import QuarantineSanitizer

        self.sanitizer = QuarantineSanitizer(self.cfg, detector=self.detector)
        self.rules_sanitizer = QuarantineSanitizer(self.cfg, detector=None, load_slm=False)
        slm = "SLM fallback on" if self.sanitizer.rewriter else "SLM fallback off (strip)"
        return f"mode={self.cfg['layer2']['mode']} · spaCy {'on' if self.sanitizer.nlp else 'off'} · {slm}"

    def _load_queries(self) -> dict[str, dict]:
        path = Path(self.cfg["paths"]["processed_dir"]) / "rag_eval.jsonl"
        return {q["query_id"]: q for q in read_jsonl(path)} if path.exists() else {}

    def pipeline(self, name: str):
        from src.baseline_rag import UndefendedRAG
        from src.defended_rag import HybridDefendedRAG
        from src.defenses import DelimiterRAG, LLMJudgeRAG

        if self.llm is None or self.retriever is None:
            raise HTTPException(503, "LLM or vector index not loaded — see /api/status")
        if name == "undefended":
            return UndefendedRAG(self.retriever, self.llm)
        if name == "delimiter":
            return DelimiterRAG(self.retriever, self.llm)
        if name == "llm_judge":
            return LLMJudgeRAG(self.retriever, self.llm)
        if name == "hybrid":
            if self.detector is None or self.sanitizer is None:
                raise HTTPException(503, "Hybrid needs a trained Layer-1 model (python -m src.train_layer1)")
            return HybridDefendedRAG(self.retriever, self.llm, self.detector, self.sanitizer)
        raise HTTPException(400, f"unknown setup {name!r}")


# ----------------------------------------------------------------------------- API models

class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    setups: list[str] = Field(default_factory=lambda: list(SETUPS))
    query_id: str | None = None


class SanitizeIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    mode: str = "strip"
    use_detector: bool = True


def _chunk_json(c) -> dict:
    return {"chunk_id": c.chunk_id, "poisoned": bool(c.label), "flagged": c.flagged, "action": c.action,
            "confidence": c.confidence, "original": c.original_text, "delivered": c.delivered_text,
            "method": c.method, "spans": [{"start": s, "end": e, "reasons": r} for s, e, r in c.neutralized_spans]}


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Hybrid IPI Defense", version="1.0")

    @app.get("/api/status")
    def status():
        return {"components": engine.status, "n_queries": len(engine.queries), "setups": SETUPS,
                "smoke_mode": engine.cfg["llm"]["provider"] == "mock" or engine.cfg["embeddings"]["model_name"] == "hashing"}

    @app.get("/api/examples")
    def examples(n: int = 12):
        atk = [q for q in engine.queries.values() if q["is_attack"]][:n]
        ben = [q for q in engine.queries.values() if not q["is_attack"]][:n]
        keep = ("query_id", "question", "answer", "is_attack", "attack_family", "position", "payload")
        return {"attack": [{k: q.get(k) for k in keep} for q in atk], "benign": [{k: q.get(k) for k in keep} for q in ben]}

    @app.post("/api/query")
    def query(body: QueryIn):
        gold = engine.queries.get(body.query_id) if body.query_id else None
        out = []
        for name in body.setups:
            try:
                with engine.lock:
                    res = engine.pipeline(name).answer(body.question)
            except HTTPException as exc:
                out.append({"setup": name, "error": exc.detail})
                continue
            row = {"setup": name, "answer": res.answer, "chunks": [_chunk_json(c) for c in res.chunks],
                   "retrieval_ms": res.retrieval_ms, "defense_ms": res.defense_ms, "generation_ms": res.generation_ms,
                   "total_ms": res.total_ms, "prompt_tokens": res.prompt_tokens,
                   "completion_tokens": res.completion_tokens, "cost_usd": res.cost_usd, "llm_calls": res.llm_calls,
                   "layer1_ms": res.extra.get("layer1_ms"), "layer2_ms": res.extra.get("layer2_ms")}
            # Hijack check works for free-typed questions too: any retrieved payload's canary.
            canaries = {c.canary for c in res.chunks if c.label == 1 and c.canary}
            if gold and gold.get("canary"):
                canaries.add(gold["canary"])
            row["attack_success"] = any(canary_hit(res.answer, k) for k in canaries) if canaries else None
            if gold:
                row["correct"] = answer_correct(res.answer, gold["answer"])
            out.append(row)
        return {"question": body.question, "gold": gold and {k: gold.get(k) for k in
                ("answer", "is_attack", "attack_family", "position", "canary")}, "results": out}

    @app.post("/api/sanitize")
    def sanitize(body: SanitizeIn):
        if engine.rules_sanitizer is None:
            raise HTTPException(503, engine.status.get("layer2", {}).get("detail", "Layer 2 not loaded"))
        san = engine.sanitizer if (body.use_detector and engine.detector is not None) else engine.rules_sanitizer
        with engine.lock:
            prev_mode, san.mode = san.mode, body.mode if body.mode in ("strip", "describe") else "strip"
            try:
                verdicts = san.analyze(body.text)
                res = san.sanitize(body.text)
                chunk_score = engine.detector.detect([body.text])[0] if engine.detector is not None else None
            finally:
                san.mode = prev_mode
        return {
            "sentences": [{"text": v.text, "start": v.start, "end": v.end, "score": v.rule_score, "reasons": v.reasons,
                           "detector_prob": v.detector_prob, "flagged": v.flagged} for v in verdicts],
            "sanitized": res.sanitized, "method": res.method, "latency_ms": res.latency_ms,
            "residual_score": res.residual_score, "used_detector": san is engine.sanitizer,
            "layer1": chunk_score and {"is_suspicious": chunk_score.is_suspicious,
                                       "confidence": chunk_score.confidence_score, "latency_ms": chunk_score.latency_ms},
        }

    @app.get("/api/results")
    def results():
        rd = Path(engine.cfg["paths"]["results_dir"])

        def read_csv(name):
            p = rd / name
            return list(csv.DictReader(p.open())) if p.exists() else []

        fp = rd / "fact_preservation.json"
        return {"comparison": read_csv("comparison.csv"), "asr_by_family": read_csv("asr_by_family.csv"),
                "fact_preservation": json.loads(fp.read_text()) if fp.exists() else None,
                "results_dir": str(rd)}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--provider", choices=["openai", "ollama", "mock"], default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args()
    engine = Engine(load_config(args.config), args.provider, args.rebuild_index)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
