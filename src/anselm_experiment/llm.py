"""Thin LLM client wrapper.

Wraps litellm so every call goes through one place that:
- pins the model name into the run log
- counts tokens (prompt + completion)
- records the full request/response verbatim under runs/<timestamp>/
- enforces a single retry policy

Replace placeholder bodies before running real experiments.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# litellm is imported lazily inside call() so the module imports without keys set.


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class CallResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLM:
    def __init__(
        self,
        model: str | None = None,
        log_dir: Path | None = None,
        temperature: float = 0.0,
        seed: int | None = 42,
        progress: bool = True,
    ) -> None:
        self.model = model or os.environ.get("ANSELM_DEFAULT_MODEL", "gpt-4o-mini")
        self.log_dir = log_dir
        self.temperature = temperature
        self.seed = seed
        self.progress = progress
        self._call_index = 0

    def call(
        self,
        messages: list[Message],
        *,
        tag: str = "untagged",
        temperature: float | None = None,
        seed: int | None = None,
    ) -> CallResult:
        """Make a single LLM call. Records request/response if log_dir is set.

        `temperature` and `seed` override the instance defaults for this call only.
        """
        import litellm

        self._call_index += 1

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
        }
        effective_seed = self.seed if seed is None else seed
        if effective_seed is not None:
            kwargs["seed"] = effective_seed

        if self.progress:
            import sys
            t0 = time.time()
            print(f"  · call#{self._call_index:02d} [{tag}] … ", end="", flush=True)
        response = litellm.completion(**kwargs)
        if self.progress:
            dt = time.time() - t0
            print(f"{dt:5.1f}s", flush=True)
        # litellm normalises responses to OpenAI-style dicts.
        raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        choice = raw["choices"][0]["message"]
        usage = raw.get("usage") or {}

        result = CallResult(
            content=choice.get("content") or "",
            model=raw.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw=raw,
        )

        if self.log_dir is not None:
            self._record(messages, result, tag)
        return result

    def _record(self, messages: list[Message], result: CallResult, tag: str) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"call_{self._call_index:04d}_{tag}.json"
        payload = {
            "ts": time.time(),
            "model": result.model,
            "tag": tag,
            "messages": [m.__dict__ for m in messages],
            "response": {
                "content": result.content,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
