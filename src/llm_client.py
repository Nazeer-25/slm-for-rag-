"""Provider-agnostic chat client (OpenAI / Ollama / offline mock) with token
accounting, cost estimation and an on-disk response cache.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1, so a
single code path serves both providers. The defense never needs model internals:
this client is the only interface to the main LLM (black-box assumption).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field

from src.utils import get_logger

log = get_logger("llm_client")


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool = False
    model: str = ""


@dataclass
class UsageMeter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    by_tag: dict = field(default_factory=dict)

    def add(self, r: LLMResponse, tag: str = "generation") -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cost_usd += r.cost_usd
        self.latency_ms += r.latency_ms
        t = self.by_tag.setdefault(tag, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        t["calls"] += 1
        t["prompt_tokens"] += r.prompt_tokens
        t["completion_tokens"] += r.completion_tokens
        t["cost_usd"] += r.cost_usd


class _Cache:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()


@functools.lru_cache(maxsize=1)
def _encoder():
    """Load tiktoken once; None if unavailable (it downloads its BPE file on first use)."""
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001
        log.warning("tiktoken encoding unavailable — approximating tokens as len(text)/4")
        return None


def _approx_tokens(text: str) -> int:
    enc = _encoder()
    return len(enc.encode(text)) if enc else max(1, len(text) // 4)


class LLMClient:
    def __init__(self, cfg: dict, provider: str | None = None, model: str | None = None):
        lcfg = cfg["llm"]
        self.provider = provider or os.environ.get("IPI_LLM_PROVIDER") or lcfg["provider"]
        self.temperature = lcfg["temperature"]
        self.max_tokens = lcfg["max_tokens"]
        self.pricing = lcfg["pricing"]
        self.cache = _Cache(cfg["paths"]["cache_db"]) if lcfg.get("use_cache", True) else None
        if self.provider == "openai":
            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set (put it in .env or export it)")
            self.model = model or lcfg["openai_model"]
            self._client = OpenAI(timeout=lcfg["timeout_s"], max_retries=3)
        elif self.provider == "ollama":
            from openai import OpenAI

            self.model = model or os.environ.get("OLLAMA_MODEL") or lcfg["ollama_model"]
            base = os.environ.get("OLLAMA_BASE_URL") or lcfg["ollama_base_url"]
            self._client = OpenAI(base_url=base, api_key="ollama", timeout=lcfg["timeout_s"] * 5, max_retries=2)
        elif self.provider == "mock":
            self.model = "mock-vulnerable-llm"
            self._client = None
        else:
            raise ValueError(f"unknown provider {self.provider!r}")

    def _price(self, pt: int, ct: int) -> float:
        if self.provider != "openai":
            return 0.0
        p = self.pricing.get(self.model, self.pricing["default"])
        return (pt * p["input"] + ct * p["output"]) / 1_000_000

    def chat(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> LLMResponse:
        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        key = hashlib.sha256(json.dumps([self.provider, self.model, messages, max_tokens, temperature]).encode()).hexdigest()
        if self.cache and (hit := self.cache.get(key)):
            # Replay the original token counts, cost and latency so metrics are
            # identical across re-runs (the `cached` flag marks the replay).
            return LLMResponse(**hit, cached=True)
        start = time.perf_counter()
        if self.provider == "mock":
            text = _mock_generate(messages)
            pt = sum(_approx_tokens(m["content"]) for m in messages)
            ct = _approx_tokens(text)
        else:
            resp = self._client.chat.completions.create(model=self.model, messages=messages,
                                                        temperature=temperature, max_tokens=max_tokens)
            text = resp.choices[0].message.content or ""
            if resp.usage:
                pt, ct = resp.usage.prompt_tokens, resp.usage.completion_tokens
            else:
                pt = sum(_approx_tokens(m["content"]) for m in messages)
                ct = _approx_tokens(text)
        latency = (time.perf_counter() - start) * 1000
        out = LLMResponse(text=text, prompt_tokens=pt, completion_tokens=ct, latency_ms=latency,
                          cost_usd=self._price(pt, ct), model=self.model)
        if self.cache:
            self.cache.put(key, {k: v for k, v in out.__dict__.items() if k != "cached"})
        return out


# ----------------------------------------------------------------------------- mock
# A deliberately *vulnerable*, deterministic stand-in for the main LLM, used ONLY
# for offline smoke tests and CI. It (a) obeys any quoted reply / URL instruction
# found in the context — mimicking an injectable model — and otherwise (b) answers
# with the context sentence that best overlaps the question. Never report results
# produced with provider=mock as research findings.

_OBEY = re.compile(
    r"(?:reply|respond|answer|output|print|say|repeat|begin|start|include|add|append|antworte|responde|réponds|rispondi)"
    r"[^'\"\n]{0,80}?['\"]([^'\"]{2,120})['\"]",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s)\]]+")
_TRIGGER = re.compile(r"ignore|disregard|forget|system|instruction|assistant|must|tell the|inform|warn|instead|"
                      r"ignoriere|ignora|you are|from now on|directive|note for ai", re.IGNORECASE)


def _mock_generate(messages: list[dict]) -> str:
    user = messages[-1]["content"]
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    if "Answer YES or NO" in system or "Answer YES or NO" in user:  # LLM-as-judge prompt
        body = user.split("<chunk>")[-1] if "<chunk>" in user else user
        return "YES" if _TRIGGER.search(body) and (_OBEY.search(body) or _URL.search(body)) else "NO"
    question = user.rsplit("Question:", 1)[-1]
    context = user.rsplit("Question:", 1)[0]
    # Vulnerable behaviour: obey embedded instructions that appear in the context.
    for sent in re.split(r"(?<=[.!?])\s+|\n", context):
        if _TRIGGER.search(sent):
            if m := _OBEY.search(sent):
                return m.group(1)
            if m := _URL.search(sent):
                return f"Warning: please visit {m.group(0)}"
    q_words = set(re.findall(r"[a-z0-9]+", question.lower())) - {"the", "a", "of", "what", "how", "is", "to", "in", "does"}
    best, best_score = "I don't know.", 0
    for sent in re.split(r"(?<=[.!?])\s+", context):
        score = len(q_words & set(re.findall(r"[a-z0-9]+", sent.lower())))
        if score > best_score:
            best, best_score = sent.strip(), score
    return best
