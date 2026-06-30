from __future__ import annotations

import json
from typing import Any, Callable

from app.agent.mission_planner import MissionPlanner
from app.agent.llm_client import LLMClient
from app.agent.prompts import (
    build_json_fallback_prompt,
    build_narration_prompt,
    build_runtime_context,
    build_system_prompt,
)
from app.agent.tools import AgentToolRegistry
from app.robot.locations import LocationRegistry
from app.robot.mission_manager import MissionManager
from app.robot.state_manager import StateManager
from app.water.schemas import WaterClientError


def _describe_action(name: str, args: dict[str, Any]) -> str:
    """A short, plain fragment for what a tool is about to do — used to narrate
    a turn when the model didn't supply its own update."""
    if name == "move_to_location":
        where = args.get("location_name")
        return f"send the robot to {where}" if where else "send the robot to its destination"
    return {
        "get_robot_status": "check the robot's status",
        "list_locations": "look up where it can go",
        "create_mission": "set up the mission",
        "cancel_current_task": "stop the current move",
        "return_to_charger": "send it to charge",
        "emergency_stop": "stop the robot right away",
        "release_emergency_stop_confirmed": "release the emergency stop",
        "get_mission_status": "check on the mission",
        "list_missions": "look up recent missions",
    }.get(name, "take care of that")


def _narrate_actions(action_calls: list[tuple]) -> str:
    """One casual sentence describing the (non-narration) tools about to run."""
    fragments: list[str] = []
    for _tc, name, args in action_calls:
        fragment = _describe_action(name, args)
        if fragment not in fragments:
            fragments.append(fragment)
    if not fragments:
        return "On it."
    if len(fragments) == 1:
        joined = fragments[0]
    elif len(fragments) == 2:
        joined = f"{fragments[0]} and {fragments[1]}"
    else:
        joined = ", ".join(fragments[:-1]) + f", and {fragments[-1]}"
    return f"I'll {joined}."


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
        # Only surface missions that are still in flight — finished/failed ones
        # are durable history, not live context, and shouldn't resurface (e.g.
        # after a restart) and confuse the agent.
        active = {"created", "running", "waiting"}
        missions = [m for m in self.mission_manager.list_missions() if m["status"] in active][:3]
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

    def _model_narration(self, message: str, action_calls: list[tuple]) -> str | None:
        """Ask the model for a one-line 'what I'm about to do' update, with Qwen
        thinking turned off so it stays fast. Returns None on any failure (the
        caller then falls back to a deterministic template), so a slow or down
        model never blocks the turn for long."""
        descriptions = "; ".join(_describe_action(name, args) for _tc, name, args in action_calls)
        try:
            text = self.llm_client.complete(
                build_narration_prompt(message, descriptions), max_tokens=60, timeout=20.0
            )
        except Exception:  # noqa: BLE001 - narration is best-effort; fall back
            return None
        if not text:
            return None
        # Keep just the first line and strip any stray quotes the model adds.
        line = text.strip().splitlines()[0].strip().strip('"').strip()
        return line or None

    def run_chat(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        summary: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        # on_event (optional) streams the turn as it happens: a {"type": ...}
        # event per narration / tool call / tool result, so the UI can render
        # live instead of waiting for the whole turn. None -> silent (tests,
        # offline/JSON-fallback paths).
        def emit(event: dict[str, Any]) -> None:
            if on_event is not None:
                on_event(event)

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
        ]
        # Prior conversation: a summary of compacted turns (if any) then the
        # retained recent turns, before the new user message.
        if summary:
            messages.append({"role": "system", "content": f"Summary of earlier conversation: {summary}"})
        messages.extend(history or [])
        messages.append({"role": "user", "content": message})

        # Cap the number of rounds that actually DO something (read/move/mission)
        # so a turn can't loop forever; status_update rounds are pure narration
        # and don't count against the budget.
        action_iterations = 0
        total_rounds = 0
        while action_iterations < 4 and total_rounds < 8:
            total_rounds += 1
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
                    "streamed": on_event is not None,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            # Pre-parse the round so we can guarantee a short "what I'm about to
            # do" line BEFORE the tool calls — using the model's own words when
            # it offered them (a content preamble or status_update), otherwise a
            # deterministic line built from the tools themselves. The local model
            # often skips status_update, so we never rely on it.
            parsed: list[tuple[dict, str, dict]] = []
            narrations: list[str] = []
            if content.strip():
                narrations.append(content.strip())
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments") or "{}"
                parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                if name == "status_update":
                    text = str(parsed_arguments.get("message", "")).strip()
                    if text:
                        narrations.append(text)
                parsed.append((tool_call, name, parsed_arguments))

            action_calls = [(tc, n, a) for (tc, n, a) in parsed if n != "status_update"]
            if narrations:
                for text in narrations:
                    emit({"type": "narration", "text": text})
            elif action_calls:
                # Blocking: generate the update line and emit it BEFORE any
                # tool-call event, so the tool call never appears ahead of the
                # message describing it (a little latency is acceptable).
                text = self._model_narration(message, action_calls) if on_event is not None else None
                emit({"type": "narration", "text": text or _narrate_actions(action_calls)})

            did_action = False
            for tool_call, name, parsed_arguments in parsed:
                if name == "status_update":
                    # Narration already surfaced above; just ack it to the model
                    # and don't record it as a tool call or count it as an action.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", name),
                            "name": name,
                            "content": json.dumps({"ack": True}),
                        }
                    )
                    continue

                did_action = True
                emit({"type": "tool_call", "name": name, "arguments": parsed_arguments})
                try:
                    execution = self.tool_registry.execute(name, parsed_arguments)
                    payload = execution.payload
                    created_task_ids.extend(execution.task_ids)
                    created_mission_ids.extend(execution.mission_ids)
                except Exception as exc:  # surface tool errors back to the model instead of aborting
                    payload = {"error": str(exc)}
                emit({"type": "tool_result", "name": name, "payload": payload})
                tool_calls_used.append({"name": name, "arguments": parsed_arguments, "payload": payload})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", name),
                        "name": name,
                        "content": json.dumps(payload),
                    }
                )

            if did_action:
                action_iterations += 1

        return {
            "assistant_response": "Reached the tool execution limit for this turn.",
            "tool_calls": tool_calls_used,
            "created_task_ids": created_task_ids,
            "created_mission_ids": created_mission_ids,
            "final_robot_state": self.state_manager.get_compact_robot_state(),
            "streamed": on_event is not None,
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
