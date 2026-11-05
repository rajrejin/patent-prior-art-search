"""LLM interface. Backends: Ollama (default) or llama-cpp-python. Never a hosted API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """The local model server cannot be reached or the model is not installed."""


@dataclass
class LLMResult:
    text: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    backend: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 700,
    ) -> LLMResult: ...

    def available(self) -> bool: ...


def generate_structured(
    llm: LLMClient,
    schema: type[T],
    prompt: str,
    *,
    system: str | None = None,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int = 700,
    attempts: int = 2,
) -> tuple[T, LLMResult]:
    """Ask for JSON matching `schema` and validate it with Pydantic, retrying on bad output."""
    js = json_schema or schema.model_json_schema()
    last_err = ""
    for _ in range(attempts):
        extra = (
            f"\n\nYour previous answer was invalid ({last_err}). Return valid JSON only."
            if last_err
            else ""
        )
        res = llm.complete(prompt + extra, system=system, json_schema=js, max_tokens=max_tokens)
        try:
            return schema.model_validate(json.loads(res.text)), res
        except (json.JSONDecodeError, ValidationError) as exc:
            last_err = type(exc).__name__
    raise ValueError(f"LLM did not return valid {schema.__name__} after {attempts} attempts")
