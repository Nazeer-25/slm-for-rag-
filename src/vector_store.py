"""One persistent Chroma database (data/chroma) with three collections:

  enterprise_policies  the 200 synthetic chunks (clean and poisoned variants, filtered by metadata)
  external_examples    imported reference examples, when data/external_dataset.csv exists
  live_payloads        temporary custom documents for a single run, deleted in `finally`

Vectors are always computed here (MiniLM) and passed to Chroma explicitly. Chroma never
embeds anything itself; its built-in default would silently download a different model.
Only identifiers and retrieval filters are stored as metadata: labels, clean references and
attacker targets stay in the CSVs and are joined in later, for display and scoring only.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from config import ENTERPRISE_COLLECTION, EXTERNAL_COLLECTION, LIVE_COLLECTION, PATHS, Paths

ENTERPRISE_FIELDS = ("pair_id", "division", "topic", "variant")
EXTERNAL_FIELDS = ("source",)


class VectorStore:
    def __init__(self, embedder, paths: Paths = PATHS):
        import chromadb
        from chromadb.config import Settings

        paths.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.paths = paths
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(paths.chroma_dir),
                                                settings=Settings(anonymized_telemetry=False))

    # ------------------------------------------------------------------ collections
    def _collection(self, name: str):
        return self.client.get_or_create_collection(name=name, embedding_function=None,
                                                    metadata={"hnsw:space": "cosine"})

    def counts(self) -> dict[str, int | None]:
        existing = {c if isinstance(c, str) else c.name for c in self.client.list_collections()}
        return {name: (self._collection(name).count() if name in existing else None)
                for name in (ENTERPRISE_COLLECTION, EXTERNAL_COLLECTION, LIVE_COLLECTION)}

    def _manifest(self) -> dict:
        try:
            return json.loads(self.paths.index_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.paths.index_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def ensure_index(self, name: str, rows: list[dict], fields: tuple[str, ...]) -> dict:
        """(Re)build a collection only when the CSV content or the embedding model changed."""
        start = time.perf_counter()
        digest = hashlib.sha256(getattr(self.embedder, "name", "embedder").encode())
        for r in rows:
            digest.update(f"{r['chunk_id']}\t{r['text']}\n".encode("utf-8"))
        fingerprint = digest.hexdigest()
        manifest = self._manifest()
        collection = self._collection(name)
        if manifest.get(name) == fingerprint and collection.count() == len(rows):
            return {"collection": name, "count": collection.count(), "rebuilt": False,
                    "ms": (time.perf_counter() - start) * 1000}
        self.client.delete_collection(name)
        collection = self._collection(name)
        for i in range(0, len(rows), 64):
            batch = rows[i:i + 64]
            collection.add(
                ids=[r["chunk_id"] for r in batch],
                documents=[r["text"] for r in batch],
                metadatas=[{f: str(r.get(f, "")) for f in fields} for r in batch],
                embeddings=self.embedder.embed([r["text"] for r in batch]),
            )
        manifest[name] = fingerprint
        self._save_manifest(manifest)
        return {"collection": name, "count": collection.count(), "rebuilt": True,
                "ms": (time.perf_counter() - start) * 1000}

    def drop(self, name: str) -> None:
        existing = {c if isinstance(c, str) else c.name for c in self.client.list_collections()}
        if name in existing:
            self.client.delete_collection(name)
        manifest = self._manifest()
        if manifest.pop(name, None) is not None:
            self._save_manifest(manifest)

    # ------------------------------------------------------------------ retrieval
    def query(self, name: str, vector: list[float], top_k: int, where: dict | None = None) -> list[dict]:
        collection = self._collection(name)
        if collection.count() == 0:
            return []
        res = collection.query(query_embeddings=[vector], n_results=max(1, min(top_k, collection.count())),
                               where=where or None, include=["documents", "metadatas", "distances"])
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({"chunk_id": cid, "collection": name, "text": doc, "metadata": meta or {},
                         "distance": float(dist), "similarity": 1.0 - float(dist)})
        return hits

    # ------------------------------------------------------------------ live payloads
    def add_live(self, run_id: str, texts: list[str]) -> list[str]:
        ids = [f"live-{run_id}-{i}" for i in range(len(texts))]
        self._collection(LIVE_COLLECTION).add(ids=ids, documents=texts, metadatas=[{"run_id": run_id}] * len(texts),
                                              embeddings=self.embedder.embed(texts))
        return ids

    def delete_live(self, ids: list[str]) -> None:
        if ids:
            self._collection(LIVE_COLLECTION).delete(ids=ids)

    def purge_live(self) -> int:
        """Remove leftovers from a run that was interrupted (e.g. the window was closed)."""
        collection = self._collection(LIVE_COLLECTION)
        ids = collection.get(include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)


def chroma_counts_readonly(paths: Paths = PATHS) -> dict[str, int | None]:
    """Collection counts without loading an embedder (for status pages)."""
    if not Path(paths.chroma_dir).is_dir():
        return {ENTERPRISE_COLLECTION: None, EXTERNAL_COLLECTION: None, LIVE_COLLECTION: None}
    return VectorStore(embedder=None, paths=paths).counts()
