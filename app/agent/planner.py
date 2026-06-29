from __future__ import annotations

import json
from typing import Any

from app.agent.mission_planner import MissionPlanner
from app.agent.llm_client import LLMClient
from app.agent.prompts import build_json_fallback_prompt, build_runtime_context, build_system_prompt
from app.agent.tools import AgentToolRegistry
from app.robot.locations import LocationRegistry
from app.robot.mission_manager import MissionManager
from app.robot.state_manager import StateManager
from app.water.schemas import WaterClientError


class AgentPlanner:
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: AgentToolRegistry,
        state_manager: StateManager,
        location_registry: LocationRegistry,
        mission_manager: MissionManager,
        mission_planner: MissionPlanner,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.state_manager = state_manager
        self.location_registry = location_registry
        self.mission_manager = mission_manager
        self.mission_planner = mission_planner

    def _compact_locations(self) -> dict[str, Any]:
        locations = self.location_registry.list_locations()
        return {
            "markers": [item["marker_name"] for item in locations["markers"][:40]],
            "aliases": locations["aliases"][:25],
        }

    def _compact_missions(self) -> dict[str, Any]:
        missions = self.mission_manager.list_missions()[:3]
        return {
            "recent_missions": [
                {
                    "mission_id": mission["mission_id"],
                    "status": mission["status"],
                    "current_step_index": mission["current_step_index"],
                    "summary": mission["context_summary"],
                }
                for mission in missions
            ]
        }

    def run_chat(self, message: str) -> dict[str, Any]:
        # Let create_mission record the originating request.
        self.tool_registry.current_message = message

        settings = self.llm_client.settings
        # No live tool-calling model available: fall back to the phrasing
        # heuristic (the only way to act without an LLM).
        if not settings.llm_enabled or settings.llm_dry_run:
            return self._run_offline(message)

        # Single tool-calling loop. The model decides everything — including
        # whether a request is a durable multi-step mission (create_mission tool)
        # versus a single move or a status question. No keyword routing.
        tool_calls_used: list[dict[str, Any]] = []
        created_task_ids: list[str] = []
        created_mission_ids: list[str] = []
        state_summary = self.state_manager.get_compact_robot_state()
        locations = self._compact_locations()
        missions = self._compact_missions()
        identity = self.state_manager.get_device_identity(refresh=True)
        chargers = self.location_registry.list_chargers(identity)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "system", "content": build_runtime_context(state_summary, locations, missions, identity, chargers)},
            {"role": "user", "content": message},
        ]

        for _ in range(4):
            try:
                response = self.llm_client.chat_with_tools(messages, self.tool_registry.definitions())
            except Exception:
                return self._run_json_fallback(message)

            choice = response.get("choices", [{}])[0]
            assistant_message = choice.get("message", {})
            tool_calls = assistant_message.get("tool_calls") or []
            content = assistant_message.get("content") or ""

            if not tool_calls:
                return {
                    "assistant_response": content or "No action taken.",
                    "tool_calls": tool_calls_used,
                    "created_task_ids": created_task_ids,
                    "created_mission_ids": created_mission_ids,
                    "final_robot_state": self.state_manager.get_compact_robot_state(),
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments") or "{}"
                parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                try:
                    execution = self.tool_registry.execute(name, parsed_arguments)
                    payload = execution.payload
                    created_task_ids.extend(execution.task_ids)
                    created_mission_ids.extend(execution.mission_ids)
                except Exception as exc:  # surface tool errors back to the model instead of aborting
                    payload = {"error": str(exc)}
                tool_calls_used.append({"name": name, "arguments": parsed_arguments, "payload": payload})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", name),
                        "name": name,
                        "content": json.dumps(payload),
                    }
                )

        return {
            "assistant_response": "Reached the tool execution limit for this turn.",
            "tool_calls": tool_calls_used,
            "created_task_ids": created_task_ids,
            "created_mission_ids": created_mission_ids,
            "final_robot_state": self.state_manager.get_compact_robot_state(),
        }

    def _run_offline(self, message: str) -> dict[str, Any]:
        """No-LLM path: build a mission from the phrasing heuristic if possible."""
        steps = self.mission_planner.heuristic_steps(message)
        if steps:
            try:
                mission = self.mission_manager.create_mission(
                    user_request=message,
                    steps=steps,
                    name="Planned mission",
                    auto_replan=True,
                )
            except ValueError as exc:
                return self._simple_response(f"Could not build a mission: {exc}")
            return {
                "assistant_response": "Created a mission plan.",
                "tool_calls": [],
                "created_task_ids": [],
                "created_mission_ids": [mission["mission_id"]],
                "final_robot_state": self.state_manager.get_compact_robot_state(),
                "mission": mission,
            }
        return self._simple_response(
            "The language model is unavailable, so I can only act on explicit step commands "
            "(e.g. 'go to front desk, then return to charger')."
        )

    def _simple_response(self, text: str) -> dict[str, Any]:
        return {
            "assistant_response": text,
            "tool_calls": [],
            "created_task_ids": [],
            "created_mission_ids": [],
            "final_robot_state": self.state_manager.get_compact_robot_state(),
        }

    def _run_json_fallback(self, message: str) -> dict[str, Any]:
        # Used when the model errors or doesn't support tool calling. Returns a
        # single action; on its own failure, drops to the offline heuristic.
        state = self.state_manager.get_compact_robot_state()
        locations = self._compact_locations()
        missions = self._compact_missions()
        identity = self.state_manager.get_device_identity()
        chargers = self.location_registry.list_chargers(identity)
        prompt = build_json_fallback_prompt(message, state, locations, missions, identity, chargers)
        try:
            plan = self.llm_client.json_plan(prompt)
        except WaterClientError:
            return self._run_offline(message)
        tool_calls_used: list[dict[str, Any]] = []
        created_task_ids: list[str] = []
        created_mission_ids: list[str] = []
        if plan.get("action"):
            action = plan["action"]
            try:
                execution = self.tool_registry.execute(action["tool"], action.get("arguments", {}))
                payload = execution.payload
                created_task_ids.extend(execution.task_ids)
                created_mission_ids.extend(execution.mission_ids)
            except Exception as exc:  # noqa: BLE001
                payload = {"error": str(exc)}
            tool_calls_used.append({"name": action["tool"], "arguments": action.get("arguments", {}), "payload": payload})
        return {
            "assistant_response": plan.get("response", "Done."),
            "tool_calls": tool_calls_used,
            "created_task_ids": created_task_ids,
            "created_mission_ids": created_mission_ids,
            "final_robot_state": self.state_manager.get_compact_robot_state(),
        }
