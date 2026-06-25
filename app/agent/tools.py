from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.robot.locations import LocationRegistry
from app.robot.mission_manager import MissionManager
from app.robot.state_manager import StateManager
from app.robot.task_manager import TaskManager


@dataclass
class ToolExecution:
    content: str
    task_ids: list[str]
    payload: dict[str, Any]


class AgentToolRegistry:
    def __init__(
        self,
        state_manager: StateManager,
        location_registry: LocationRegistry,
        task_manager: TaskManager,
        mission_manager: MissionManager,
    ):
        self.state_manager = state_manager
        self.location_registry = location_registry
        self.task_manager = task_manager
        self.mission_manager = mission_manager

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_robot_status",
                    "description": "Get a compact summary of the robot state.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_locations",
                    "description": "List available robot markers and aliases.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "move_to_location",
                    "description": "Move the robot to a known marker or alias by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location_name": {"type": "string"},
                            "allow_interruption": {"type": "boolean"},
                        },
                        "required": ["location_name"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_current_task",
                    "description": "Cancel the robot's current movement task.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "return_to_charger",
                    "description": "Send the robot to the configured charging marker.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "emergency_stop",
                    "description": "Immediately place the robot into emergency stop.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "release_emergency_stop_confirmed",
                    "description": "Release emergency stop only after explicit confirmation.",
                    "parameters": {
                        "type": "object",
                        "properties": {"confirmed": {"type": "boolean"}},
                        "required": ["confirmed"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_mission_status",
                    "description": "Fetch the latest status for a mission by mission_id.",
                    "parameters": {
                        "type": "object",
                        "properties": {"mission_id": {"type": "string"}},
                        "required": ["mission_id"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_missions",
                    "description": "List recent durable missions and their statuses.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        if name == "get_robot_status":
            payload = self.state_manager.get_compact_robot_state()
            return ToolExecution(content="Robot status fetched.", task_ids=[], payload=payload)
        if name == "list_locations":
            payload = self.location_registry.list_locations()
            return ToolExecution(content="Available locations fetched.", task_ids=[], payload=payload)
        if name == "move_to_location":
            task = self.task_manager.create_move_marker_task(
                arguments["location_name"],
                allow_interruption=bool(arguments.get("allow_interruption", False)),
            )
            return ToolExecution(content=f"Created move task {task['task_id']}.", task_ids=[task["task_id"]], payload=task)
        if name == "cancel_current_task":
            task = self.task_manager.cancel_current_move()
            return ToolExecution(content=f"Created cancel task {task['task_id']}.", task_ids=[task["task_id"]], payload=task)
        if name == "return_to_charger":
            task = self.task_manager.return_to_charger()
            return ToolExecution(content=f"Created charger task {task['task_id']}.", task_ids=[task["task_id"]], payload=task)
        if name == "emergency_stop":
            task = self.task_manager.emergency_stop()
            return ToolExecution(content=f"Created emergency stop task {task['task_id']}.", task_ids=[task["task_id"]], payload=task)
        if name == "release_emergency_stop_confirmed":
            task = self.task_manager.release_emergency_stop(bool(arguments.get("confirmed", False)))
            return ToolExecution(content=f"Created estop release task {task['task_id']}.", task_ids=[task["task_id"]], payload=task)
        if name == "get_mission_status":
            payload = self.mission_manager.get_mission(arguments["mission_id"])
            return ToolExecution(content=f"Fetched mission {arguments['mission_id']}.", task_ids=[], payload=payload)
        if name == "list_missions":
            payload = {"missions": self.mission_manager.list_missions()}
            return ToolExecution(content="Fetched mission list.", task_ids=[], payload=payload)
        raise ValueError(f"Unsupported tool '{name}'.")
