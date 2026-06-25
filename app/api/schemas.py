from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MoveMarkerRequest(BaseModel):
    marker_name: str
    allow_interruption: bool = False


class MoveLocationRequest(BaseModel):
    x: float
    y: float
    theta: float
    allow_interruption: bool = False


class ReleaseEstopRequest(BaseModel):
    confirmed: bool = Field(default=False)


class LocationAliasRequest(BaseModel):
    alias: str
    marker_name: str


class AgentChatRequest(BaseModel):
    message: str


class MissionStepInput(BaseModel):
    step_type: Literal["move_marker", "move_coordinate", "wait", "return_to_charger", "cancel_move", "emergency_stop"]
    marker_name: str | None = None
    x: float | None = None
    y: float | None = None
    theta: float | None = None
    wait_seconds: int | None = None
    allow_interruption: bool = False
    description: str | None = None


class MissionCreateRequest(BaseModel):
    user_request: str
    steps: list[MissionStepInput]
    name: str | None = None
    auto_replan: bool = False


class MissionPlanRequest(BaseModel):
    message: str
    auto_replan: bool = True
