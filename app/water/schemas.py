from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WaterEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "response"
    command: str
    status: str
    error_message: str = ""
    uuid: str = ""
    results: Any = None
    task_id: str | None = Field(default=None, alias="task_id")


class WaterClientError(Exception):
    """Raised when the robot API cannot be reached or parsed."""
