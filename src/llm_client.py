"""Generation backends. Default: a local Ollama model (no API key, no charges).

Optional remote provider: set LLM_PROVIDER=openai and OPENAI_API_KEY (and optionally
OPENAI_MODEL). That provider may incur API charges.

The defense treats the model as a black box: these classes are its only interface.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import config


class LLMError(RuntimeError):
    """A generation request failed (server unreachable, model missing, timeout, ...)."""


@dataclass
class LLMReply:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    model: str
    provider: str


class OllamaClient:
    provider = "ollama"

    def __init__(self, host: str = config.OLLAMA_HOST, model: str = config.OLLAMA_MODEL,
                 timeout: float = config.LLM_TIMEOUT_S, max_tokens: int = config.LLM_MAX_TOKENS):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        # Never route localhost traffic through a system or corporate proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def label(self) -> str:
        return f"{self.model} via Ollama"

    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.host + path, data=data, method="GET" if data is None else "POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                message = json.loads(body).get("error", body)
            except ValueError:
                message = body
            if exc.code == 404 and "not found" in message.lower():
                raise LLMError(f"Ollama does not have the model '{self.model}'. Run: ollama pull {self.model}") from exc
            raise LLMError(f"Ollama returned HTTP {exc.code}: {message[:300]}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LLMError(f"Ollama did not answer within {timeout or self.timeout:.0f} s "
                           "(the first load of a model can be slow; try again).") from exc
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise LLMError(f"Ollama did not answer within {timeout or self.timeout:.0f} s "
                               "(the first load of a model can be slow; try again).") from exc
            raise LLMError(f"Cannot reach Ollama at {self.host} ({reason}). Open the Ollama app, or run "
                           f"'ollama serve', then check with: ollama run {self.model} \"Reply with READY.\"") from exc

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> LLMReply:
        payload = {"model": self.model, "messages": messages, "stream": False, "keep_alive": "30m",
                   "options": {"temperature": config.LLM_TEMPERATURE, "seed": config.LLM_SEED,
                               "num_predict": max_tokens or self.max_tokens}}
        start = time.perf_counter()
        data = self._request("/api/chat", payload)
        latency = (time.perf_counter() - start) * 1000
        if data.get("error"):
            raise LLMError(f"Ollama error: {data['error']}")
        text = (data.get("message") or {}).get("content", "")
        return LLMReply(text.strip(), latency, int(data.get("prompt_eval_count") or 0),
                        int(data.get("eval_count") or 0), self.model, self.provider)

    def health(self) -> dict:
        try:
            tags = self._request("/api/tags", timeout=3)
        except LLMError as exc:
            return {"reachable": False, "model_present": False, "models": [], "error": str(exc)}
        names = [m.get("name", "") for m in tags.get("models", [])]
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return {"reachable": True, "model_present": wanted in names or self.model in names, "models": names,
                "error": None if (wanted in names or self.model in names) else
                f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}"}

    def warm_up(self) -> LLMReply:
        """The same smoke test as `ollama run <model> "Reply with READY."`: proves real generation works."""
        return self.chat([{"role": "user", "content": "Reply with READY."}], max_tokens=8)


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str = config.OPENAI_MODEL, timeout: float = config.LLM_TIMEOUT_S,
                 max_tokens: int = config.LLM_MAX_TOKENS):
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError("LLM_PROVIDER=openai needs OPENAI_API_KEY to be set.")
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(timeout=timeout, max_retries=2)

    @property
    def label(self) -> str:
        return f"{self.model} via OpenAI API"

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> LLMReply:
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(model=self.model, messages=messages,
                                                        temperature=config.LLM_TEMPERATURE, seed=config.LLM_SEED,
                                                        max_tokens=max_tokens or self.max_tokens)
        except Exception as exc:  # noqa: BLE001 - surface any SDK error as a failed run
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        usage = resp.usage
        return LLMReply((resp.choices[0].message.content or "").strip(), (time.perf_counter() - start) * 1000,
                        usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0,
                        self.model, self.provider)

    def health(self) -> dict:
        return {"reachable": True, "model_present": True, "models": [self.model], "error": None}

    def warm_up(self) -> LLMReply:
        return self.chat([{"role": "user", "content": "Reply with READY."}], max_tokens=8)


def make_llm_client(provider: str | None = None):
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "openai":
        return OpenAIClient()
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}' (use 'ollama' or 'openai').")


def describe_llm(provider: str | None = None) -> str:
    """Model label without connecting (for status pages)."""
    provider = (provider or config.LLM_PROVIDER).lower()
    return f"{config.OPENAI_MODEL} via OpenAI API" if provider == "openai" else f"{config.OLLAMA_MODEL} via Ollama"
