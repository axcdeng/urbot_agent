from __future__ import annotations

import json
import re
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


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

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
            response = httpx.post(url, headers=self._headers(), json=payload, timeout=self.settings.llm_timeout_seconds)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WaterClientError(f"Tool-calling request failed: {exc}") from exc

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
            response = httpx.post(url, headers=self._headers(), json=payload, timeout=self.settings.llm_timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WaterClientError(f"Fallback planning request failed: {exc}") from exc
        content = data["choices"][0]["message"].get("content", "{}")
        return extract_json_object(content)
