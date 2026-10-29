"""Offline test doubles. Tests never download models or call a real LLM.

* HashEmbedder      deterministic bag-of-words vectors in place of MiniLM
* ScriptedLLM       a deliberately vulnerable stand-in for the main LLM (obeys planted codes)
* FakeOllamaServer  a real HTTP server speaking Ollama's API, so the real client code is exercised
* build_tiny_distilbert  a small randomly initialised DistilBERT with a vocabulary learned from the corpus

The website and CLIs never use these; they exist only under tests/.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import Paths  # noqa: E402
import data_generator  # noqa: E402

TARGET_RE = re.compile(r"\b(?:SR|AD|MEMO)-\d{4}\b|\b555-\d{4}\b|\b(?:verify|collect)-\d{4}\.example\.(?:com|net)\b")


def temp_workspace(with_data: bool = True) -> Paths:
    root = Path(tempfile.mkdtemp(prefix="workbench-test-"))
    paths = Paths(root)
    shutil.copytree(ROOT / "web", root / "web")
    if with_data:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            data_generator.main([], paths=paths)
    return paths


def remove_workspace(paths: Paths) -> None:
    shutil.rmtree(paths.root, ignore_errors=True)


class HashEmbedder:
    name = "test-hash-embedder"
    dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            words = re.findall(r"[a-z0-9]+", text.lower())
            for tok in words + [a + "_" + b for a, b in zip(words, words[1:])]:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class ScriptedLLM:
    """Vulnerable test model: repeats a planted attacker code if one reaches the prompt,
    otherwise answers with the context sentence that mentions a number."""
    provider = "scripted"
    model = "scripted-test-llm"
    label = "scripted test LLM"

    def __init__(self, fail: bool = False):
        self.calls: list[list[dict]] = []
        self.fail = fail

    def chat(self, messages, max_tokens=None):
        from llm_client import LLMError, LLMReply

        self.calls.append(messages)
        if self.fail:
            raise LLMError("scripted failure")
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        user = messages[-1]["content"]
        if "YES or NO" in system + user:
            text = "YES" if re.search(r"assistant|system|ignore", user, re.I) else "NO"
        elif "Reply with READY" in user:
            text = "READY"
        elif m := TARGET_RE.search(user):
            text = f"Notice: {m.group(0)}"
        else:
            body, _, question = re.sub(r"\[\d+\]", "", user).rpartition("Question:")
            q_words = set(re.findall(r"[a-z]+", question.lower())) - {"what", "is", "the", "how", "do", "does"}
            numbered = [s for s in re.split(r"(?<=[.!?])\s+|\n", body) if re.search(r"\d", s)]
            # answer from the numbered sentence of the document that best overlaps the question
            docs = [d for d in re.split(r"\n\s*\n|</?document[^>]*>", body) if d.strip()]
            best_doc = max(docs, key=lambda d: len(q_words & set(re.findall(r"[a-z]+", d.lower()))), default="")
            in_best = [s for s in numbered if s.strip() and s.strip() in best_doc]
            text = (in_best or numbered or ["I don't know."])[0].strip()
        return LLMReply(text, 1.0, len(user) // 4, len(text) // 4, self.model, self.provider)

    def health(self):
        return {"reachable": True, "model_present": True, "models": [self.model], "error": None}

    def warm_up(self):
        return self.chat([{"role": "user", "content": "Reply with READY."}])


class FakeOllamaServer:
    """Speaks /api/tags and /api/chat. `reply` may be a string or a callable(messages) -> str."""

    def __init__(self, models=("llama3.2:3b",), reply="Customers may return items within 90 days.",
                 status: int = 200, error: str | None = None):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in server.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(payload)
                if server.status != 200:
                    self._send(server.status, {"error": server.error or "failure"})
                    return
                text = server.reply(payload["messages"]) if callable(server.reply) else server.reply
                self._send(200, {"model": payload["model"], "message": {"role": "assistant", "content": text},
                                 "done": True, "prompt_eval_count": 42, "eval_count": 7})

        self.models, self.reply, self.status, self.error = list(models), reply, status, error
        self.requests: list[dict] = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def build_tiny_distilbert(out_dir: Path, texts: list[str]) -> Path:
    """A 2-layer DistilBERT (random weights) + WordPiece vocabulary trained on `texts`."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(
        vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]", model_max_length=512)
    out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out_dir))
    cfg = DistilBertConfig(vocab_size=fast.vocab_size, dim=64, hidden_dim=128, n_layers=2, n_heads=2,
                           max_position_embeddings=512, num_labels=2)
    DistilBertForSequenceClassification(cfg).save_pretrained(str(out_dir))
    return out_dir


_TRAINED_CACHE: Path | None = None


def trained_workspace() -> Paths:
    """A fresh temp workspace with data and a trained tiny detector (trained once per test run)."""
    global _TRAINED_CACHE
    import contextlib
    import io

    paths = temp_workspace()
    if _TRAINED_CACHE is None or not _TRAINED_CACHE.exists():
        import train_layer1

        base = build_tiny_distilbert(paths.root / "tiny-base", [r["text"] for r in data_generator.read_csv(paths.dataset_csv)])
        with contextlib.redirect_stdout(io.StringIO()):
            train_layer1.main(["--base-model", str(base), "--epochs", "8", "--lr", "1e-3"], paths=paths)
        _TRAINED_CACHE = Path(tempfile.mkdtemp(prefix="workbench-detector-")) / "layer1_detector"
        shutil.copytree(paths.detector_dir, _TRAINED_CACHE)
        shutil.copy2(paths.detector_metrics_json, _TRAINED_CACHE.parent / "detector_metrics.json")
    else:
        shutil.copytree(_TRAINED_CACHE, paths.detector_dir)
        paths.results_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_TRAINED_CACHE.parent / "detector_metrics.json", paths.detector_metrics_json)
    return paths


def offline_workspace(paths: Paths, llm=None):
    """Workspace wired to the offline stand-ins (real Chroma, real spaCy, tiny detector)."""
    from workspace import Workspace

    return Workspace(paths, embedder_factory=HashEmbedder, llm_factory=lambda: llm or ScriptedLLM())
