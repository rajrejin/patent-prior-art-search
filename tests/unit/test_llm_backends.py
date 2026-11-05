"""LLM backends: HTTP behaviour of the Ollama client with a mock transport (no model needed)."""

from __future__ import annotations

import json

import httpx
import pytest

from patentrag.config import Secrets, load_config
from patentrag.llm.client import LLMUnavailable
from patentrag.llm.llamacpp_backend import LlamaCppClient
from patentrag.llm.ollama_backend import OllamaClient


def _client(handler, monkeypatch) -> OllamaClient:
    monkeypatch.setattr("patentrag.llm.ollama_backend.time.sleep", lambda s: None)
    secrets = Secrets(ollama_host="http://ollama.test")
    return OllamaClient(load_config(), secrets, transport=httpx.MockTransport(handler))


def test_complete_sends_schema_seed_and_temperature(monkeypatch):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={"message": {"content": '{"a": 1}'}, "prompt_eval_count": 7, "eval_count": 3},
        )

    c = _client(handler, monkeypatch)
    res = c.complete("hi", system="sys", json_schema={"type": "object"}, max_tokens=12)
    assert res.text == '{"a": 1}' and res.prompt_tokens == 7 and res.completion_tokens == 3
    assert seen["format"] == {"type": "object"}
    assert seen["options"]["temperature"] == 0.0 and seen["options"]["seed"] == 0
    assert seen["options"]["num_predict"] == 12
    assert seen["messages"][0] == {"role": "system", "content": "sys"}


def test_complete_retries_transient_errors_then_succeeds(monkeypatch):
    replies = iter([httpx.Response(500), httpx.Response(200, json={"message": {"content": "ok"}})])
    c = _client(lambda req: next(replies), monkeypatch)
    assert c.complete("hi").text == "ok"


def test_complete_raises_unavailable_after_repeated_failures(monkeypatch):
    c = _client(lambda req: httpx.Response(503), monkeypatch)
    with pytest.raises(LLMUnavailable):
        c.complete("hi")


def test_available_checks_the_model_is_installed(monkeypatch):
    model = load_config().llm.model
    c = _client(lambda req: httpx.Response(200, json={"models": [{"name": model}]}), monkeypatch)
    assert c.available()
    c2 = _client(
        lambda req: httpx.Response(200, json={"models": [{"name": "other:1b"}]}), monkeypatch
    )
    assert not c2.available()


def test_llamacpp_fallback_reports_unavailable_without_model(monkeypatch):
    monkeypatch.delenv("PATENTRAG_GGUF", raising=False)
    c = LlamaCppClient(load_config(), gguf_path="")
    assert not c.available()
    with pytest.raises(LLMUnavailable):
        c.complete("hi")
