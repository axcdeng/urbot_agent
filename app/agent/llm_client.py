from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.water.schemas import WaterClientError


# Reasoning models such as Qwen3 emit <think>...</think> blocks and frequently
# wrap structured output in ```json ... ``` fences. The local mlx_lm.server does
# not strip these, so we normalize the assistant content before parsing JSON.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json_object(content: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output.

    Tolerates leading whitespace, <think> reasoning blocks, and markdown code
    fences by isolating the substring between the first '{' and last '}'.
    """
    cleaned = _THINK_RE.sub("", content or "")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise WaterClientError("Planner response contained no JSON object.")
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise WaterClientError("Planner returned invalid JSON.") from exc


@dataclass
class LLMMetrics:
    """Accumulated LLM usage for one chat turn (may span several calls)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    # Prompt tokens of the most recent call — the fullest single context we
    # sent this turn (a tool loop makes several calls; the last is the biggest).
    # Used as the context-fullness gauge, vs cumulative prompt_tokens above.
    last_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tps(self) -> float:
        # End-to-end generation rate: output tokens over wall-clock spent in
        # the model calls (includes prompt processing, so a conservative TPS).
        return self.completion_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.metrics = LLMMetrics()

    def reset_metrics(self) -> None:
        self.metrics = LLMMetrics()

    def _record(self, data: dict[str, Any], elapsed: float) -> None:
        usage = (data or {}).get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.metrics.calls += 1
        self.metrics.prompt_tokens += prompt_tokens
        self.metrics.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.metrics.elapsed_s += elapsed
        if prompt_tokens:
            self.metrics.last_prompt_tokens = prompt_tokens

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.llm_api_key}"}

    def chat_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.settings.llm_enabled:
            return {"choices": [{"message": {"content": "LLM is disabled.", "tool_calls": []}}]}
        if self.settings.llm_dry_run:
            return {"choices": [{"message": {"content": "Dry-run assistant response.", "tool_calls": []}}]}
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.settings.llm_max_tokens,
        }
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        try:
            t0 = time.perf_counter()
            response = httpx.post(url, headers=self._headers(), json=payload, timeout=self.settings.llm_timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WaterClientError(f"Tool-calling request failed: {exc}") from exc
        self._record(data, time.perf_counter() - t0)
        return data

    def json_plan(self, prompt: str) -> dict[str, Any]:
        if not self.settings.llm_enabled:
            return {"response": "LLM is disabled.", "action": None}
        if self.settings.llm_dry_run:
            return {"response": "Dry-run fallback response.", "action": None}
        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.settings.llm_max_tokens,
        }
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        try:
            t0 = time.perf_counter()
            response = httpx.post(url, headers=self._headers(), json=payload, timeout=self.settings.llm_timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WaterClientError(f"Fallback planning request failed: {exc}") from exc
        self._record(data, time.perf_counter() - t0)
        content = data["choices"][0]["message"].get("content", "{}")
        return extract_json_object(content)

    def complete(self, prompt: str, max_tokens: int | None = None) -> str:
        """Plain-text completion for short utilities (chat titles, summaries).

        Returns the assistant text with any <think> reasoning block stripped.
        Raises WaterClientError on transport failure; callers fall back when the
        LLM is disabled/dry-run (both return an empty string here).
        """
        if not self.settings.llm_enabled or self.settings.llm_dry_run:
            return ""
        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        try:
            t0 = time.perf_counter()
            response = httpx.post(url, headers=self._headers(), json=payload, timeout=self.settings.llm_timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WaterClientError(f"Completion request failed: {exc}") from exc
        self._record(data, time.perf_counter() - t0)
        content = data["choices"][0]["message"].get("content", "") or ""
        return _THINK_RE.sub("", content).strip()
