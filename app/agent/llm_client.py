from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import Settings
from app.water.schemas import WaterClientError


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
        }
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        try:
            response = httpx.post(url, headers=self._headers(), json=payload, timeout=30.0)
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
        }
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        try:
            response = httpx.post(url, headers=self._headers(), json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WaterClientError(f"Fallback planning request failed: {exc}") from exc
        content = data["choices"][0]["message"].get("content", "{}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise WaterClientError("Fallback planner returned invalid JSON.") from exc
