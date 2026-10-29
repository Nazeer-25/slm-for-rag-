import os
import socket
import unittest
from unittest import mock

import _testkit as tk
import config
from llm_client import LLMError, OllamaClient, make_llm_client


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OllamaClientTest(unittest.TestCase):
    def test_chat_sends_a_deterministic_non_streaming_request(self):
        with tk.FakeOllamaServer(reply="Refunds are accepted within 90 days.") as srv:
            client = OllamaClient(host=srv.url, model="llama3.2:3b")
            reply = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.text, "Refunds are accepted within 90 days.")
        self.assertEqual((reply.prompt_tokens, reply.completion_tokens, reply.provider), (42, 7, "ollama"))
        sent = srv.requests[0]
        self.assertEqual(sent["model"], "llama3.2:3b")
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["options"]["temperature"], 0.0)
        self.assertIn("seed", sent["options"])

    def test_warm_up_is_the_ready_smoke_test(self):
        with tk.FakeOllamaServer(reply="READY") as srv:
            reply = OllamaClient(host=srv.url).warm_up()
        self.assertEqual(reply.text, "READY")
        self.assertEqual(srv.requests[0]["messages"][-1]["content"], "Reply with READY.")

    def test_health_reports_a_missing_model(self):
        with tk.FakeOllamaServer(models=["qwen2.5:0.5b"]) as srv:
            health = OllamaClient(host=srv.url, model="llama3.2:3b").health()
        self.assertTrue(health["reachable"])
        self.assertFalse(health["model_present"])
        self.assertIn("ollama pull llama3.2:3b", health["error"])

    def test_missing_model_error_says_how_to_pull_it(self):
        with tk.FakeOllamaServer(status=404, error="model 'llama3.2:3b' not found, try pulling it first") as srv:
            with self.assertRaises(LLMError) as ctx:
                OllamaClient(host=srv.url).chat([{"role": "user", "content": "hi"}])
        self.assertIn("ollama pull", str(ctx.exception))

    def test_unreachable_server_error_is_actionable(self):
        client = OllamaClient(host=f"http://127.0.0.1:{free_port()}", timeout=3)
        with self.assertRaises(LLMError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("Cannot reach Ollama", str(ctx.exception))
        self.assertFalse(client.health()["reachable"])

    def test_ollama_host_forms_are_normalized(self):
        n = config._normalize_ollama_host
        self.assertEqual(n("0.0.0.0"), "http://127.0.0.1:11434")
        self.assertEqual(n("127.0.0.1:11434"), "http://127.0.0.1:11434")
        self.assertEqual(n("http://localhost"), "http://localhost:11434")
        self.assertEqual(n("http://127.0.0.1:11434/"), "http://127.0.0.1:11434")

    def test_openai_provider_requires_a_key(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises(LLMError):
                make_llm_client("openai")
        with self.assertRaises(LLMError):
            make_llm_client("something-else")


if __name__ == "__main__":
    unittest.main()
