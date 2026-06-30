from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.mission_planner import MissionPlanner
from app.robot.locations import LocationRegistry
from app.robot.mission_manager import MissionManager
from app.robot.state_manager import StateManager
from app.robot.task_manager import TaskManager


@dataclass
class ToolExecution:
    content: str
    task_ids: list[str]
    payload: dict[str, Any]
    mission_ids: list[str] = field(default_factory=list)


class AgentToolRegistry:
    def __init__(
        self,
        state_manager: StateManager,
        location_registry: LocationRegistry,
        task_manager: TaskManager,
        mission_manager: MissionManager,
        mission_planner: MissionPlanner,
    ):
        self.state_manager = state_manager
        self.location_registry = location_registry
        self.task_manager = task_manager
        self.mission_manager = mission_manager
        self.mission_planner = mission_planner
        # Set by the planner before each turn so create_mission can record the
        # originating user request.
        self.current_message: str | None = None

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "status_update",
                    "description": (
                        "Tell the user, in ONE short plain present-tense sentence, what you are "
                        "about to do — call this RIGHT BEFORE the tool call(s) it describes (and "
                        "once at the very start of a turn to acknowledge, e.g. 'On it.'). This is "
                        "the only way the user sees what you're doing, since your reasoning is "
                        "hidden. It performs no robot action. Keep it casual; no IDs, no jargon, "
                        "no emojis."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                },
            },
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
                    "description": (
                        "Move the robot to ONE known marker or alias by name. Use this for a single "
                        "destination. If the destination is a charging dock that does not serve this "
                        "robot, the move is blocked and asks for confirmation; only then re-call with "
                        "confirm=true after the user explicitly agrees."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location_name": {"type": "string"},
                            "allow_interruption": {"type": "boolean"},
                            "confirm": {
                                "type": "boolean",
                                "description": "Set true ONLY after the user confirms moving to a charger that is not this robot's own.",
                            },
                        },
                        "required": ["location_name"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_mission",
                    "description": (
                        "Create a durable, ordered multi-step mission. Use this whenever the user "
                        "asks for a SEQUENCE of actions (e.g. go somewhere, wait, then go elsewhere, "
                        "then return to charger) instead of issuing several move_to_location calls. "
                        "Missions are durable and recover on their own if a step fails. Use only "
                        "known markers/aliases. "
                        "IMPORTANT: a mission STARTS RUNNING the moment it is created — the robot "
                        "begins the first step immediately. There is no separate 'start' action and "
                        "you do NOT need to ask the user whether to begin."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "description": "Ordered steps to execute.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step_type": {
                                            "type": "string",
                                            "enum": ["move_marker", "wait", "return_to_charger", "cancel_move", "emergency_stop"],
                                        },
                                        "marker_name": {"type": "string", "description": "Required for move_marker; a known marker or alias."},
                                        "wait_seconds": {"type": "integer", "description": "Required for wait."},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["step_type"],
                                },
                            },
                            "mission_name": {"type": "string", "description": "Short title for the mission."},
                        },
                        "required": ["steps"],
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
        if name == "status_update":
            # Narration only: surfaced to the user by the planner's event stream;
            # performs no robot action and returns a trivial ack to the model.
            return ToolExecution(content="ok", task_ids=[], payload={"ack": True})
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
                confirm_foreign_charger=bool(arguments.get("confirm", False)),
            )
            return ToolExecution(content=f"Created move task {task['task_id']}.", task_ids=[task["task_id"]], payload=task)
        if name == "create_mission":
            steps = self.mission_planner.normalize_steps(arguments.get("steps") or [])
            mission = self.mission_manager.create_mission(
                user_request=self.current_message or "agent mission",
                steps=steps,
                name=arguments.get("mission_name"),
                auto_replan=True,
            )
            return ToolExecution(
                content=f"Created mission {mission['mission_id']}.",
                task_ids=[],
                payload=mission,
                mission_ids=[mission["mission_id"]],
            )
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
