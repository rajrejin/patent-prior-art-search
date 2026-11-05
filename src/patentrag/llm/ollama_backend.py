"""Ollama backend (local server, JSON-schema constrained decoding)."""

from __future__ import annotations

import time
from typing import Any

import httpx

from patentrag.config import Config, Secrets, load_config, load_secrets
from patentrag.llm.client import LLMResult, LLMUnavailable


class OllamaClient:
    backend = "ollama"

    def __init__(
        self,
        cfg: Config | None = None,
        secrets: Secrets | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 600.0,
    ):
        cfg = cfg or load_config()
        secrets = secrets or load_secrets()
        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        self.num_ctx = cfg.llm.num_ctx
        self.seed = cfg.seed
        self.host = secrets.ollama_host.rstrip("/")
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def available(self) -> bool:
        try:
            r = self._http.get(f"{self.host}/api/tags", timeout=5)
            names = {m.get("name", "") for m in r.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return False
        return any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 700,
    ) -> LLMResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        if json_schema is not None:
            body["format"] = json_schema
        t0 = time.perf_counter()
        last: Exception | None = None
        data: dict[str, Any] = {}
        for attempt in range(4):  # transient server errors are retried before giving up
            try:
                r = self._http.post(f"{self.host}/api/chat", json=body)
                r.raise_for_status()
                data = r.json()
                break
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                time.sleep(5 * (attempt + 1))
        else:
            raise LLMUnavailable(f"Ollama request failed ({type(last).__name__})") from last
        return LLMResult(
            text=data["message"]["content"],
            seconds=time.perf_counter() - t0,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
        )
