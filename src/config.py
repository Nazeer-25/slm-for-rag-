"""Central configuration: paths, model names, defense settings and the LLM provider.

Every setting below can be overridden with an environment variable, or with a
``KEY=VALUE`` line in an optional ``.env`` file in the project root. On Windows:

    set OLLAMA_MODEL=llama3.2:3b
    set LLM_PROVIDER=openai          (optional remote provider; may incur API charges)

Paths are grouped in :class:`Paths` so tests can point a whole workspace at a
temporary directory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (no dependency). Real environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _normalize_ollama_host(raw: str) -> str:
    """Accept the forms Ollama itself accepts ("0.0.0.0", "127.0.0.1:11434", URLs)."""
    host = raw.strip().rstrip("/")
    if "://" not in host:
        host = "http://" + host
    scheme, rest = host.split("://", 1)
    if rest.startswith("0.0.0.0"):  # a bind address, not something a client can call
        rest = "127.0.0.1" + rest[len("0.0.0.0"):]
    if ":" not in rest.split("/", 1)[0]:
        rest = rest.replace("/", ":11434/", 1) if "/" in rest else rest + ":11434"
    return f"{scheme}://{rest}"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def dataset_csv(self) -> Path:
        return self.data_dir / "dataset.csv"

    @property
    def train_csv(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_dir / "test.csv"

    @property
    def external_csv(self) -> Path:
        return self.data_dir / "external_dataset.csv"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def index_manifest(self) -> Path:
        return self.chroma_dir / "index_manifest.json"

    @property
    def history_json(self) -> Path:
        return self.data_dir / "demo_history.json"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def detector_dir(self) -> Path:
        return self.models_dir / "layer1_detector"

    @property
    def embedding_cache(self) -> Path:
        return self.models_dir / "embedding_cache"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def detector_metrics_json(self) -> Path:
        return self.results_dir / "detector_metrics.json"

    @property
    def benchmark_runs_json(self) -> Path:
        return self.results_dir / "benchmark_runs.json"

    @property
    def benchmark_results_json(self) -> Path:
        return self.results_dir / "benchmark_results.json"

    @property
    def benchmark_chart_png(self) -> Path:
        return self.results_dir / "benchmark_comparison.png"

    @property
    def web_dir(self) -> Path:
        return self.root / "web"

    def dataset_files(self) -> dict[str, Path]:
        return {"dataset": self.dataset_csv, "train": self.train_csv, "test": self.test_csv,
                "external": self.external_csv}


PATHS = Paths(PROJECT_ROOT)

# --------------------------------------------------------------------------- models
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
DETECTOR_BASE_MODEL = _env("DETECTOR_BASE_MODEL", "distilbert-base-uncased")
DETECTOR_THRESHOLD = 0.50       # route to Layer 2 at P(injection) >= 0.50
DETECTOR_MAX_LENGTH = 256       # tokens per window
DETECTOR_STRIDE = 64            # overlap between consecutive windows
SPACY_MODEL = "en_core_web_sm"

# --------------------------------------------------------------------------- vector store
ENTERPRISE_COLLECTION = "enterprise_policies"
EXTERNAL_COLLECTION = "external_examples"
LIVE_COLLECTION = "live_payloads"
DEFAULT_TOP_K = int(_env("TOP_K", "3"))

# --------------------------------------------------------------------------- defense modes
MODES = ("no_defense", "delimiter_defense", "llm_judge_defense", "hybrid_defense")
MODE_LABELS = {
    "no_defense": "No defense",
    "delimiter_defense": "Delimiter defense",
    "llm_judge_defense": "LLM-as-judge",
    "hybrid_defense": "Hybrid defense (L1 + L2)",
}
MODE_DESCRIPTIONS = {
    "no_defense": "Retrieved chunks are pasted into the prompt unchanged.",
    "delimiter_defense": "Chunks are wrapped in <context> tags and the model is told to treat them as data.",
    "llm_judge_defense": "The LLM screens every retrieved chunk first; chunks it flags are excluded.",
    "hybrid_defense": "Layer 1 scores every chunk; flagged chunks go to Layer 2, which removes "
                      "assistant-directed clauses and keeps the facts.",
}

CORPORA = {
    "enterprise_clean": {"label": "Enterprise policies - clean", "collection": ENTERPRISE_COLLECTION,
                         "where": {"variant": "clean"}},
    "enterprise_poisoned": {"label": "Enterprise policies - poisoned", "collection": ENTERPRISE_COLLECTION,
                            "where": {"variant": "poisoned"}},
    "external": {"label": "External reference examples", "collection": EXTERNAL_COLLECTION, "where": None},
}

HISTORY_LIMIT = 30

# --------------------------------------------------------------------------- LLM
LLM_PROVIDER = _env("LLM_PROVIDER", "ollama").lower()
OLLAMA_HOST = _normalize_ollama_host(_env("OLLAMA_HOST", "http://127.0.0.1:11434"))
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2:3b")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_S = float(_env("LLM_TIMEOUT_S", "300"))
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "160"))
LLM_TEMPERATURE = 0.0
LLM_SEED = 42

# --------------------------------------------------------------------------- website
WEB_HOST = _env("WORKBENCH_HOST", "127.0.0.1")
WEB_PORT = int(_env("WORKBENCH_PORT", "8000"))
