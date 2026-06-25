from __future__ import annotations


def build_system_prompt() -> str:
    return (
        "You are a robot task planner.\n"
        "You may only use the provided tools.\n"
        "Never invent marker names.\n"
        "Prefer marker-based navigation.\n"
        "If a location is unknown or ambiguous, ask for clarification.\n"
        "Do not use raw WATER API commands.\n"
        "Do not use direct velocity control.\n"
        "Safety validation is handled by the backend.\n"
        "Keep user-facing responses short and clear."
    )


def build_runtime_context(state_summary: dict, locations: dict, missions: dict | None = None) -> str:
    mission_line = f"Mission context: {missions}\n" if missions else ""
    return f"Robot state: {state_summary}\nAvailable locations: {locations}\n{mission_line}"


def build_json_fallback_prompt(user_message: str, state_summary: dict, locations: dict, missions: dict | None = None) -> str:
    return (
        "Respond with JSON only.\n"
        "Schema:\n"
        '{"response":"short user-facing reply","action":{"tool":"tool_name","arguments":{}}}\n'
        'If no tool is needed, set "action" to null.\n'
        "Allowed tools: get_robot_status, list_locations, move_to_location, cancel_current_task, return_to_charger, emergency_stop, release_emergency_stop_confirmed, get_mission_status, list_missions.\n"
        f"{build_runtime_context(state_summary, locations, missions)}"
        f"User message: {user_message}\n"
    )
