"""Shared helpers: config loading, seeding, JSONL I/O, timing, logging."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("IPI_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load YAML config; relative paths in `paths` are resolved against the project root."""
    load_env()
    path = Path(path or os.environ.get("IPI_CONFIG") or DEFAULT_CONFIG)
    cfg = _read_yaml_with_extends(path)
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
        if key.endswith("_dir"):
            Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    for section in ("layer1", "layer1_qlora"):
        out = Path(cfg[section]["output_dir"])
        cfg[section]["output_dir"] = str(out if out.is_absolute() else PROJECT_ROOT / out)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml_with_extends(path: Path) -> dict[str, Any]:
    """A config may declare `extends: other.yaml` (relative to itself) and override any keys."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(_read_yaml_with_extends((path.parent / parent).resolve()), cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@contextmanager
def timer() -> Iterator[dict]:
    """Usage: `with timer() as t: ...; t["ms"]`."""
    box: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = (time.perf_counter() - start) * 1000.0
