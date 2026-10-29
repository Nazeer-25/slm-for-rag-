"""all-MiniLM-L6-v2 sentence embeddings (384 dimensions), cached in models/embedding_cache.

The first load downloads the model from Hugging Face. After that, `set HF_HUB_OFFLINE=1`
makes every later start load it from the cache without any network access.
"""
from __future__ import annotations

from pathlib import Path

from config import EMBEDDING_MODEL, PATHS


def embedding_cached(cache_dir: Path = PATHS.embedding_cache, model_name: str = EMBEDDING_MODEL) -> bool:
    short = model_name.split("/")[-1]
    return cache_dir.is_dir() and any(short in p.name for p in cache_dir.iterdir())


class MiniLMEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_dir: Path = PATHS.embedding_cache,
                 device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir), device=device)
        except Exception as exc:  # noqa: BLE001 - turn a cryptic hub error into an actionable one
            hint = ("The embedding model is not cached yet: connect to the internet once (and unset "
                    "HF_HUB_OFFLINE) so it can download into models/embedding_cache."
                    if not embedding_cached(cache_dir, model_name) else "")
            raise RuntimeError(f"Could not load {model_name}: {exc}. {hint}".strip()) from exc
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True)
        return vectors.astype("float32").tolist()
