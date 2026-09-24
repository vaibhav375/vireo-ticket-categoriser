"""Local-LLM path (vireo/classify.py) against a fake Ollama server: memory guards and failure handling.

A real HTTP server on localhost stands in for Ollama, so the real request code runs.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from vireo import classify


class FakeOllama:
    def __init__(self, models, loaded=(), chat=None):
        self.models, self.loaded, self.chat = models, list(loaded), chat or (lambda msg: {"category": "Connectivity"})
        self.unloaded, self.chat_calls = [], 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode() if not isinstance(body, bytes) else body
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": n, "size": s} for n, s in fake.models.items()]})
                elif self.path == "/api/ps":
                    self._send(200, {"models": [{"name": n} for n in fake.loaded]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/generate" and body.get("keep_alive") == 0:
                    fake.unloaded.append(body["model"])
                    self._send(200, {"done": True})
                elif self.path == "/api/chat":
                    fake.chat_calls += 1
                    reply = fake.chat(body["messages"][-1]["content"])
                    if isinstance(reply, int):
                        self._send(reply, {"error": "boom"})
                    else:
                        content = reply if isinstance(reply, str) else json.dumps(reply)
                        self._send(200, {"message": {"content": content}})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def ollama(monkeypatch):
    servers = []

    def start(**kw):
        s = FakeOllama(**kw)
        servers.append(s)
        monkeypatch.setattr(classify, "OLLAMA_URL", s.url)
        return s
    yield start
    for s in servers:
        s.close()


GB = 1_000_000_000


def tickets(confidences):
    return pd.DataFrame({"ticket_id": [f"T{i}" for i in range(len(confidences))],
                         "customer_message": [f"message {i}" for i in range(len(confidences))],
                         "ai_category": "Audio Quality", "ai_confidence": confidences, "ref_category": None})


def test_model_too_big_for_this_machine_is_refused_before_loading(ollama, monkeypatch):
    monkeypatch.setattr(classify, "_ram_bytes", lambda: 8 * GB)
    s = ollama(models={"qwen2.5:7b": int(4.7 * GB)})
    msg = classify.check_llm("qwen2.5:7b")
    assert "4.7 GB" in msg and "2.8 GB" in msg   # 35% of 8 GB
    assert s.chat_calls == 0


def test_model_that_fits_is_accepted(ollama, monkeypatch):
    monkeypatch.setattr(classify, "_ram_bytes", lambda: 8 * GB)
    ollama(models={"qwen2.5:3b": int(1.9 * GB)})
    assert classify.check_llm("qwen2.5:3b") is None


def test_ollama_not_running_is_explained(monkeypatch):
    monkeypatch.setattr(classify, "OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens here
    assert "isn't running" in classify.check_llm("qwen2.5:3b")


def test_model_not_pulled_is_explained(ollama):
    ollama(models={"llama3.2:3b": 2 * GB})
    assert "ollama pull qwen2.5:3b" in classify.check_llm("qwen2.5:3b")


def test_other_loaded_models_are_unloaded_first(ollama, monkeypatch):
    monkeypatch.setattr(classify, "_ram_bytes", lambda: 8 * GB)
    s = ollama(models={"qwen2.5:3b": 2 * GB, "qwen2.5:7b": 5 * GB}, loaded=["qwen2.5:7b"])
    classify.check_llm("qwen2.5:3b")
    assert s.unloaded == ["qwen2.5:7b"]


def test_only_low_confidence_tickets_are_sent_and_the_model_is_unloaded_after(ollama, monkeypatch):
    monkeypatch.setattr(classify, "_ram_bytes", lambda: 8 * GB)
    s = ollama(models={"qwen2.5:3b": 2 * GB})
    out, usage = classify.llm_review(tickets([0.05, 0.9, 0.1]))
    assert s.chat_calls == 2
    assert out.ai_category.tolist() == ["Connectivity", "Audio Quality", "Connectivity"]
    assert s.unloaded == ["qwen2.5:3b"]


def test_a_failed_llm_call_keeps_the_fast_models_answer_and_the_run_continues(ollama, monkeypatch):
    monkeypatch.setattr(classify, "_ram_bytes", lambda: 8 * GB)
    replies = iter([500, {"category": "Connectivity"}])
    s = ollama(models={"qwen2.5:3b": 2 * GB}, chat=lambda msg: next(replies))
    out, usage = classify.llm_review(tickets([0.05, 0.1]))
    assert out.ai_category.tolist() == ["Audio Quality", "Connectivity"]
    assert usage["failed"] == 1
    assert s.unloaded == ["qwen2.5:3b"]


@pytest.mark.parametrize("reply", ["not json at all", {"category": "Made Up Category"}, {"wrong_key": 1}])
def test_unusable_llm_replies_keep_the_fast_models_answer(ollama, monkeypatch, reply):
    monkeypatch.setattr(classify, "_ram_bytes", lambda: 8 * GB)
    ollama(models={"qwen2.5:3b": 2 * GB}, chat=lambda msg: reply)
    out, _ = classify.llm_review(tickets([0.05]))
    assert out.ai_category.tolist() == ["Audio Quality"]
