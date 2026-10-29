"""The workspace: loads every component once and reports what is ready.

Used by the website, the Streamlit dashboard, the CLI and the benchmark. Initialization steps:

  1. datasets      data/dataset.csv, train.csv, test.csv (and external_dataset.csv if imported)
  2. embedding     all-MiniLM-L6-v2 from models/embedding_cache (downloaded on first use)
  3. vector_index  Chroma collections, rebuilt only when the CSV content changed
  4. layer1        the fine-tuned detector from models/layer1_detector
  5. layer2        spaCy en_core_web_sm + contextual rules
  6. llm           Ollama reachable, model pulled, and a real "Reply with READY." generation

Components can be swapped through the factory arguments (the tests use offline stand-ins).
"""
from __future__ import annotations

import importlib.util
import threading
import time

import config
from config import CORPORA, ENTERPRISE_COLLECTION, EXTERNAL_COLLECTION, MODES, PATHS, Paths
from embeddings import embedding_cached
from history import History
from layer1_detector import detector_available
from llm_client import describe_llm

STEP_NAMES = {
    "datasets": "Source CSV datasets",
    "embedding": "MiniLM embedding model",
    "vector_index": "Chroma vector index",
    "layer1": "Layer 1 detector",
    "layer2": "Layer 2 parser",
    "llm": "Generation model",
}


def _csv_rows(path) -> int | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig") as f:
        return max(0, sum(1 for _ in f) - 1)


def _default_embedder(paths: Paths):
    from embeddings import MiniLMEmbedder

    return MiniLMEmbedder(cache_dir=paths.embedding_cache)


def _default_llm():
    from llm_client import make_llm_client

    return make_llm_client()


def _default_detector(paths: Paths):
    from layer1_detector import Layer1Detector

    return Layer1Detector(paths.detector_dir)


def _default_sanitizer():
    from layer2_quarantine import Layer2Quarantine

    return Layer2Quarantine()


class Workspace:
    def __init__(self, paths: Paths = PATHS, embedder_factory=None, llm_factory=None, detector_factory=None,
                 sanitizer_factory=None):
        self.paths = paths
        self._factories = {
            "embedder": embedder_factory or (lambda: _default_embedder(paths)),
            "llm": llm_factory or _default_llm,
            "detector": detector_factory or (lambda: _default_detector(paths)),
            "sanitizer": sanitizer_factory or _default_sanitizer,
        }
        self.embedder = self.store = self.llm = self.detector = self.sanitizer = self.truth = None
        self.history = History(paths.history_json)
        self.steps: list[dict] = [{"name": n, "label": STEP_NAMES[n], "status": "pending", "detail": "", "ms": None}
                                  for n in STEP_NAMES]
        self.initializing = False
        self._run_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._health = (0.0, None)

    # ------------------------------------------------------------------ initialization
    def _step(self, name: str, fn) -> bool:
        step = next(s for s in self.steps if s["name"] == name)
        step.update(status="running", detail="", ms=None)
        start = time.perf_counter()
        try:
            step["detail"] = fn() or ""
            step["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - shown to the user with a fix
            step["status"], step["detail"] = "error", str(exc)
        step["ms"] = round((time.perf_counter() - start) * 1000)
        return step["status"] == "ok"

    def _skip(self, name: str, why: str) -> None:
        next(s for s in self.steps if s["name"] == name).update(status="skipped", detail=why, ms=None)

    def initialize(self, warm_up: bool = True, need_detector: bool = True) -> list[dict]:
        with self._init_lock:
            self.initializing = True
            try:
                for s in self.steps:
                    s.update(status="pending", detail="", ms=None)
                if not self._step("datasets", self._init_datasets):
                    for n in ("embedding", "vector_index", "layer1", "layer2", "llm"):
                        self._skip(n, "needs the datasets")
                    return self.steps
                ok_embed = self._step("embedding", self._init_embedding)
                if ok_embed:
                    self._step("vector_index", self._init_index)
                else:
                    self._skip("vector_index", "needs the embedding model")
                if need_detector:
                    self._step("layer1", self._init_detector)
                    self._step("layer2", self._init_sanitizer)
                else:
                    self._skip("layer1", "not needed for this mode")
                    self._skip("layer2", "not needed for this mode")
                self._step("llm", lambda: self._init_llm(warm_up))
                return self.steps
            finally:
                self.initializing = False

    def start_initialize(self) -> bool:
        """Run initialize() in the background (the website polls status())."""
        if self.initializing:
            return False
        self.initializing = True
        threading.Thread(target=self.initialize, daemon=True).start()
        return True

    def _init_datasets(self) -> str:
        from rag_pipeline import GroundTruth

        missing = [p.name for p in (self.paths.dataset_csv, self.paths.train_csv, self.paths.test_csv) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {', '.join(missing)}. Run: python src/data_generator.py")
        self.truth = GroundTruth(self.paths)
        ext = _csv_rows(self.paths.external_csv)
        return (f"dataset {_csv_rows(self.paths.dataset_csv)}, train {_csv_rows(self.paths.train_csv)}, "
                f"test {_csv_rows(self.paths.test_csv)}" + (f", external {ext}" if ext is not None else ""))

    def _init_embedding(self) -> str:
        if self.embedder is None:
            self.embedder = self._factories["embedder"]()
        return f"{self.embedder.name} ({getattr(self.embedder, 'dim', '?')} dimensions)"

    def _init_index(self) -> str:
        from data_generator import read_csv
        from vector_store import ENTERPRISE_FIELDS, EXTERNAL_FIELDS, VectorStore

        if self.store is None:
            self.store = VectorStore(self.embedder, self.paths)
        parts = []
        r = self.store.ensure_index(ENTERPRISE_COLLECTION, read_csv(self.paths.dataset_csv), ENTERPRISE_FIELDS)
        parts.append(f"{ENTERPRISE_COLLECTION} {r['count']}{' (indexed)' if r['rebuilt'] else ''}")
        if self.paths.external_csv.exists():
            r = self.store.ensure_index(EXTERNAL_COLLECTION, read_csv(self.paths.external_csv), EXTERNAL_FIELDS)
            parts.append(f"{EXTERNAL_COLLECTION} {r['count']}{' (indexed)' if r['rebuilt'] else ''}")
        else:
            self.store.drop(EXTERNAL_COLLECTION)
        purged = self.store.purge_live()
        parts.append(f"live_payloads 0" + (f" ({purged} leftovers removed)" if purged else ""))
        return ", ".join(parts)

    def _init_detector(self) -> str:
        if self.detector is None:
            if not detector_available(self.paths.detector_dir):
                raise FileNotFoundError("No trained detector in models/layer1_detector. Run: python src/train_layer1.py")
            self.detector = self._factories["detector"]()
        return f"{getattr(self.detector, 'name', 'detector')}, threshold {getattr(self.detector, 'threshold', 0.5):.2f}"

    def _init_sanitizer(self) -> str:
        if self.sanitizer is None:
            if importlib.util.find_spec(config.SPACY_MODEL) is None:
                raise ModuleNotFoundError(f"spaCy model missing. Run: python -m spacy download {config.SPACY_MODEL}")
            self.sanitizer = self._factories["sanitizer"]()
        return f"spaCy {config.SPACY_MODEL} + contextual rules"

    def _init_llm(self, warm_up: bool) -> str:
        if self.llm is None:
            self.llm = self._factories["llm"]()
        health = self.llm.health()
        self._health = (time.time(), health)
        if not health["reachable"] or not health["model_present"]:
            raise ConnectionError(health.get("error") or "LLM not reachable")
        if not warm_up:
            return f"{self.llm.label} (reachable)"
        reply = self.llm.warm_up()
        return f"{self.llm.label}: replied “{reply.text[:40]}” in {reply.latency_ms / 1000:.1f} s"

    # ------------------------------------------------------------------ status
    def available_modes(self) -> list[str]:
        if self.llm is None or self.store is None:
            return []
        return [m for m in MODES if m != "hybrid_defense" or (self.detector is not None and self.sanitizer is not None)]

    def llm_health(self, max_age_s: float = 10.0) -> dict | None:
        stamp, health = self._health
        if self.llm is not None and time.time() - stamp > max_age_s:
            health = self.llm.health()
            self._health = (time.time(), health)
        return health

    def status(self) -> dict:
        from vector_store import chroma_counts_readonly

        try:
            counts = self.store.counts() if self.store is not None else chroma_counts_readonly(self.paths)
        except Exception as exc:  # noqa: BLE001
            counts = {"error": str(exc)}
        ready = self.store is not None and self.llm is not None
        return {
            "initialized": ready,
            "initializing": self.initializing,
            "steps": self.steps,
            "available_modes": self.available_modes(),
            "corpora": {k: {"label": v["label"],
                            "available": (self.store is not None and
                                          (counts.get(v["collection"]) or 0) > 0) if isinstance(counts, dict) else False}
                        for k, v in CORPORA.items()},
            "datasets": {name: _csv_rows(path) for name, path in self.paths.dataset_files().items()},
            "collections": counts,
            "detector_trained": detector_available(self.paths.detector_dir),
            "embedding_cached": embedding_cached(self.paths.embedding_cache),
            "spacy_installed": importlib.util.find_spec(config.SPACY_MODEL) is not None,
            "llm": {"label": getattr(self.llm, "label", describe_llm()), "provider": config.LLM_PROVIDER,
                    "health": self.llm_health() if self.llm is not None else None},
            "history_count": len(self.history.all()),
        }

    # ------------------------------------------------------------------ runs
    def pipeline(self):
        from rag_pipeline import RagPipeline

        return RagPipeline(self.store, self.embedder, self.llm, self.detector, self.sanitizer)

    def run(self, question: str, mode: str, corpus: str, top_k: int = config.DEFAULT_TOP_K,
            custom_poison: str | None = None, custom_target: str | None = None, record_history: bool = True,
            expected_number: str | None = None, expected_fact: str | None = None) -> dict:
        from rag_pipeline import annotate

        if self.store is None or self.llm is None or self.truth is None:
            raise RuntimeError("The workspace is not initialized yet.")
        if mode not in self.available_modes():
            raise RuntimeError(f"Mode {mode} is not available: train Layer 1 first (python src/train_layer1.py).")
        with self._run_lock:
            trace = self.pipeline().run(question, mode, corpus, top_k, custom_poison, custom_target)
        annotate(trace, self.truth, expected_number, expected_fact)
        if record_history:
            self.history.add(trace)
        return trace
