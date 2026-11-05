"""llama-cpp-python fallback backend (used only if Ollama cannot be installed or reached)."""

from __future__ import annotations

import os
import time
from typing import Any

from patentrag.config import Config, load_config
from patentrag.llm.client import LLMResult, LLMUnavailable


class LlamaCppClient:
    backend = "llama-cpp"

    def __init__(self, cfg: Config | None = None, gguf_path: str | None = None):
        cfg = cfg or load_config()
        self.model = cfg.llm.model
        self.path = gguf_path or os.environ.get("PATENTRAG_GGUF", "")
        self.temperature = cfg.llm.temperature
        self.num_ctx = cfg.llm.num_ctx
        self.seed = cfg.seed
        self._llm: Any = None

    def available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        return bool(self.path) and os.path.exists(self.path)

    def _load(self) -> Any:
        if self._llm is None:
            if not self.available():
                raise LLMUnavailable("llama-cpp-python or the GGUF file is missing")
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=self.path, n_ctx=self.num_ctx, seed=self.seed, verbose=False
            )
        return self._llm

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 700,
    ) -> LLMResult:
        llm = self._load()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        kwargs: dict[str, Any] = {}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object", "schema": json_schema}
        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=messages, temperature=self.temperature, max_tokens=max_tokens, **kwargs
        )
        usage = out.get("usage", {})
        return LLMResult(
            text=out["choices"][0]["message"]["content"],
            seconds=time.perf_counter() - t0,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
