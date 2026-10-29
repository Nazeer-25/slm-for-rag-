"""data/demo_history.json - the most recent website experiments (full traces, newest first)."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from config import HISTORY_LIMIT


class History:
    def __init__(self, path: Path, limit: int = HISTORY_LIMIT):
        self.path = Path(path)
        self.limit = limit
        self._lock = threading.Lock()

    def all(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _write(self, runs: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(runs, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, trace: dict) -> None:
        with self._lock:
            runs = [r for r in self.all() if r.get("run_id") != trace.get("run_id")]
            self._write([trace] + runs[: self.limit - 1])

    def get(self, run_id: str) -> dict | None:
        return next((r for r in self.all() if r.get("run_id") == run_id), None)

    def clear(self) -> None:
        with self._lock:
            self._write([])

    def summaries(self) -> list[dict]:
        out = []
        for r in self.all():
            ev = r.get("evaluation") or {}
            out.append({
                "run_id": r.get("run_id"), "timestamp": r.get("timestamp"), "question": r.get("question"),
                "mode": r.get("mode"), "mode_label": r.get("mode_label"), "corpus": r.get("corpus"),
                "corpus_label": r.get("corpus_label"),
                "status": r.get("status"), "answer": r.get("answer"), "hijacked": ev.get("hijacked"),
                "fact_found": ev.get("fact_found"), "total_ms": (r.get("latency_ms") or {}).get("total"),
                "flagged": sum(bool(c.get("flagged")) for c in r.get("chunks", [])),
                "custom_poison": bool(r.get("custom_poison")),
            })
        return out
