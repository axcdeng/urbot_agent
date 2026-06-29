from __future__ import annotations

import json
import re
from typing import Any

from app.config import Settings
from app.robot.locations import LocationRegistry
from app.robot.state_manager import StateManager
from app.water.schemas import WaterClientError


SEQUENCE_PATTERNS = (" then ", " after ", " finally ", " before ", ";", " wait ")


class MissionPlanner:
    def __init__(self, llm_client, location_registry: LocationRegistry, state_manager: StateManager, settings: Settings):
        self.llm_client = llm_client
        self.location_registry = location_registry
        self.state_manager = state_manager
        self.settings = settings

    def should_plan_mission(self, message: str) -> bool:
        normalized = f" {message.strip().lower()} "
        return any(token in normalized for token in SEQUENCE_PATTERNS)

    def compact_locations(self) -> dict[str, Any]:
        locations = self.location_registry.list_locations()
        markers = [item["marker_name"] for item in locations["markers"][: self.settings.agent_max_location_names]]
        aliases = locations["aliases"][: self.settings.agent_max_aliases]
        return {"markers": markers, "aliases": aliases}

    def _normalize_step(self, step: dict[str, Any]) -> dict[str, Any]:
        step_type = step.get("step_type", "").strip().lower()
        normalized: dict[str, Any] = {"step_type": step_type}
        if step_type == "move_marker":
            normalized["marker_name"] = str(step.get("marker_name", "")).strip().lower()
            normalized["allow_interruption"] = bool(step.get("allow_interruption", False))
        elif step_type == "move_coordinate":
            normalized["x"] = float(step["x"])
            normalized["y"] = float(step["y"])
            normalized["theta"] = float(step["theta"])
            normalized["allow_interruption"] = bool(step.get("allow_interruption", False))
        elif step_type == "wait":
            normalized["wait_seconds"] = int(step.get("wait_seconds", 0))
        elif step_type in {"return_to_charger", "cancel_move", "emergency_stop"}:
            pass
        else:
            raise ValueError(f"Unsupported mission step_type '{step_type}'.")
        if step.get("description"):
            normalized["description"] = str(step["description"]).strip()
        return normalized

    def normalize_steps(self, raw_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Public normalizer for externally-supplied steps (e.g. the LLM's create_mission tool)."""
        return [self._normalize_step(step) for step in raw_steps]

    def heuristic_steps(self, message: str) -> list[dict[str, Any]]:
        """Public phrasing-based parser, used as the offline (no-LLM) fallback."""
        return self._heuristic_steps(message)

    def _heuristic_steps(self, message: str) -> list[dict[str, Any]]:
        text = message.strip().lower()
        parts = [segment.strip(" ,.") for segment in re.split(r"\bthen\b|\bafter that\b|\bfinally\b|;", text) if segment.strip(" ,.")]
        steps: list[dict[str, Any]] = []
        for part in parts:
            wait_match = re.search(r"wait(?: for)? (\d+)\s*(second|seconds|minute|minutes)", part)
            if wait_match:
                value = int(wait_match.group(1))
                unit = wait_match.group(2)
                seconds = value * 60 if "minute" in unit else value
                steps.append({"step_type": "wait", "wait_seconds": seconds, "description": f"Wait for {seconds} seconds"})
                continue
            if "return to charger" in part or "go back to charger" in part or "return to charging" in part:
                steps.append({"step_type": "return_to_charger", "description": "Return to charger"})
                continue
            if "emergency stop" in part or "estop" in part:
                steps.append({"step_type": "emergency_stop", "description": "Emergency stop"})
                continue
            move_match = re.search(r"(?:go to|move to|navigate to)\s+(.+)", part)
            if move_match:
                steps.append({"step_type": "move_marker", "marker_name": move_match.group(1).strip(), "description": f"Move to {move_match.group(1).strip()}"})
        return [self._normalize_step(step) for step in steps]

    def _build_plan_prompt(self, message: str, robot_state: dict[str, Any], locations: dict[str, Any]) -> str:
        return (
            "Respond with JSON only.\n"
            "Create a durable robot mission plan from the user's request.\n"
            "Schema:\n"
            '{"response":"short user-facing reply","mission_name":"optional short title","steps":[{"step_type":"move_marker","marker_name":"known_marker_or_alias","allow_interruption":false},{"step_type":"wait","wait_seconds":30},{"step_type":"return_to_charger"}]}\n'
            "Allowed step_type values: move_marker, move_coordinate, wait, return_to_charger, cancel_move, emergency_stop.\n"
            "Use move_marker when possible. Never invent marker names. Use only the available locations below.\n"
            f"Robot state: {robot_state}\n"
            f"Available locations: {locations}\n"
            f"User request: {message}\n"
        )

    def plan_steps_from_text(self, message: str) -> dict[str, Any]:
        robot_state = self.state_manager.get_compact_robot_state()
        locations = self.compact_locations()

        if not self.settings.llm_enabled or self.settings.llm_dry_run:
            steps = self._heuristic_steps(message)
            return {
                "response": "Created a mission plan." if steps else "I could not derive a multi-step mission from that request.",
                "mission_name": "Planned mission",
                "steps": steps,
            }

        prompt = self._build_plan_prompt(message, robot_state, locations)
        try:
            payload = self.llm_client.json_plan(prompt)
        except WaterClientError:
            steps = self._heuristic_steps(message)
            return {"response": "Created a mission plan from a compact fallback.", "mission_name": "Fallback mission", "steps": steps}

        raw_steps = payload.get("steps") or []
        steps = [self._normalize_step(step) for step in raw_steps]
        if not steps:
            steps = self._heuristic_steps(message)
        return {
            "response": payload.get("response", "Created a mission plan."),
            "mission_name": payload.get("mission_name") or "Planned mission",
            "steps": steps,
        }

    def build_replan_prompt(self, mission_context: dict[str, Any], robot_state: dict[str, Any], locations: dict[str, Any]) -> str:
        return (
            "Respond with JSON only.\n"
            "You are revising the remaining steps of a running robot mission after a problem or state change.\n"
            "Schema:\n"
            '{"response":"short user-facing reply","steps":[{"step_type":"move_marker","marker_name":"known_marker_or_alias"},{"step_type":"wait","wait_seconds":30}]}\n'
            "Return only the remaining steps needed from this point forward. Keep the plan short and safe.\n"
            "Allowed step_type values: move_marker, move_coordinate, wait, return_to_charger, cancel_move, emergency_stop.\n"
            f"Mission context: {json.dumps(mission_context, ensure_ascii=True)}\n"
            f"Robot state: {json.dumps(robot_state, ensure_ascii=True)}\n"
            f"Available locations: {json.dumps(locations, ensure_ascii=True)}\n"
        )

    def replan_steps(self, mission_context: dict[str, Any]) -> dict[str, Any] | None:
        if not self.settings.llm_enabled or self.settings.llm_dry_run:
            return None
        prompt = self.build_replan_prompt(
            mission_context,
            self.state_manager.get_compact_robot_state(),
            self.compact_locations(),
        )
        try:
            payload = self.llm_client.json_plan(prompt)
        except WaterClientError:
            return None
        try:
            steps = [self._normalize_step(step) for step in (payload.get("steps") or [])]
        except ValueError:
            return None
        if not steps:
            return None
        return {
            "response": payload.get("response", "Replanned the remaining mission steps."),
            "steps": steps,
        }
