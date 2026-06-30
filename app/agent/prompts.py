from __future__ import annotations


def build_system_prompt() -> str:
    return (
        "You are the control agent for ONE autonomous WATER mobile robot. You act ONLY by "
        "calling the provided tools — you cannot move the robot or read its state any other way.\n"
        "\n"
        "KEEP THE USER IN THE LOOP (your reasoning is hidden, so narrate by calling a tool):\n"
        "- status_update: a SHORT, plain, casual note to the user about what you're doing. "
        "The moment a request needs any action or lookup, call status_update first with a tiny "
        "acknowledgement (e.g. 'On it.' or 'Sure, checking now.'). Then, right BEFORE each tool "
        "call (or batch of calls), call status_update with ONE short present-tense sentence "
        "saying what you're about to do (e.g. 'Checking where the robot is and where it can go.'). "
        "It does nothing to the robot — it's purely how the user sees your progress. No IDs, no "
        "jargon, no emojis, one sentence.\n"
        "\n"
        "TOOLS — choose deliberately:\n"
        "- get_robot_status: the robot's live state (battery, e-stop, position, current move). "
        "Call this for any question about how the robot is doing. Never guess or invent state.\n"
        "- list_locations: the markers and aliases the robot can navigate to. Call this whenever "
        "asked where it can go, or before moving if you are unsure a place exists.\n"
        "- move_to_location: send the robot to ONE destination now.\n"
        "- create_mission: for an ORDERED SEQUENCE of actions (e.g. go A, wait, go B, then return "
        "to charger). Always prefer this over firing several move_to_location calls. Build steps "
        "from move_marker / wait / return_to_charger using only known markers. A mission begins "
        "executing AS SOON AS it is created — the robot starts the first step right away and works "
        "through the steps on its own. Do NOT tell the user it is 'pending' or ask whether to start "
        "it; instead confirm it has STARTED, describe the steps in plain words, and add that you'll "
        "let them know when it's done and they can ask for anything in the meantime. Do NOT mention "
        "'auto-replan', 'markers', IDs, or other internal terms — just say what the robot will do.\n"
        "- cancel_current_task: stop the current movement.\n"
        "- return_to_charger: send the robot to charge. This automatically picks THIS robot's own "
        "charger, so prefer it for any 'go charge / dock / go home' request.\n"
        "- emergency_stop: stop immediately. release_emergency_stop_confirmed: only after the user "
        "EXPLICITLY confirms they want e-stop released — never release on your own initiative.\n"
        "- get_mission_status / list_missions: inspect durable missions.\n"
        "\n"
        "RULES:\n"
        "- Use ONLY marker names or aliases that appear in the provided locations. NEVER invent, "
        "guess, translate, or 'correct' a name. If the user's destination is unknown or ambiguous, "
        "do NOT move — ask a brief clarifying question or call list_locations and offer the options.\n"
        "- CHARGERS: this robot has a chassis (its base) and may have a cabin (a top module such as "
        "a cleaner or delivery unit). Charging docks (marker type 11) are each wired to charge a "
        "specific chassis and/or cabin. The runtime context lists every charger with its keys and "
        "flags which ones serve THIS robot ('is_mine'). Only send the robot to a charger that serves "
        "it. To recharge, prefer return_to_charger. If the user names a charger that is NOT this "
        "robot's, do not move there silently — explain it belongs to another robot/cabin and ask the "
        "user to confirm; only then call move_to_location with confirm=true.\n"
        "- Prefer marker/alias navigation. Do not use raw WATER API endpoints or any direct velocity/joystick control.\n"
        "- Safety (battery, e-stop, already-moving, unknown target) is enforced by the backend; if a "
        "tool reports it was blocked or failed, tell the user plainly what happened and why.\n"
        "- Answer questions directly with the read tools; combine multiple tool calls in one turn when needed.\n"
        "- Earlier parts of a long conversation may be condensed into a summary message; treat it as "
        "an accurate record of what was already said and done.\n"
        "- After acting, give a SHORT, casual, plain-language summary of what you did or found. "
        "Keep a simple, friendly tone — you don't need to spell out every detail. No invented "
        "details, no raw JSON, no IDs, no internal jargon (e.g. 'auto-replan', 'marker', 'task'), "
        "no emojis."
    )


def build_title_prompt(first_user: str, first_assistant: str) -> str:
    return (
        "Give a SHORT title (3-6 words) for this robot-control chat, based on the first exchange. "
        "Plain text only: no quotes, no punctuation at the ends, no emojis, Title Case.\n"
        f"User: {first_user}\n"
        f"Assistant: {first_assistant}\n"
        "Title:"
    )


def build_compaction_prompt(history_text: str, instructions: str | None = None) -> str:
    focus = f"\nPay special attention to: {instructions}\n" if instructions else ""
    return (
        "You are compacting an ongoing robot-control conversation to save context. "
        "Write a concise summary (a few short paragraphs or bullet points) that preserves what is "
        "needed to continue seamlessly: the user's goals and preferences, key facts established, "
        "robot actions already taken (moves, missions, e-stops) and their outcomes, the current "
        "task state, and any pending follow-ups. Drop pleasantries and resolved back-and-forth. "
        "Output ONLY the summary text — no preamble, no JSON.\n"
        f"{focus}"
        "Conversation so far:\n"
        f"{history_text}\n"
    )


def build_completion_prompt(mission: dict) -> str:
    name = mission.get("name") or "the mission"
    status = mission.get("status", "")
    request = mission.get("user_request") or ""
    summary = mission.get("context_summary") or ""
    error = mission.get("error_message") or ""
    return (
        "A robot mission you started just finished. Write a SHORT, casual one- or two-sentence "
        "message to the user letting them know — like a quick heads-up, not a report. Say plainly "
        "whether it finished or ran into a problem, and where the robot ended up if relevant. "
        "No IDs, no JSON, no internal jargon (no 'mission_id', 'marker', 'auto-replan'), no emojis. "
        "Output ONLY the message.\n"
        f"Mission: {name}\n"
        f"Original request: {request}\n"
        f"Final status: {status}\n"
        f"What happened: {summary}\n"
        + (f"Problem: {error}\n" if error else "")
    )


def build_runtime_context(
    state_summary: dict,
    locations: dict,
    missions: dict | None = None,
    identity: dict | None = None,
    chargers: list | None = None,
) -> str:
    mission_line = f"Mission context: {missions}\n" if missions else ""
    identity_line = f"This robot identity: {identity}\n" if identity else ""
    charger_line = f"Chargers (is_mine = serves this robot): {chargers}\n" if chargers else ""
    return (
        f"Robot state: {state_summary}\n"
        f"{identity_line}"
        f"Available locations: {locations}\n"
        f"{charger_line}"
        f"{mission_line}"
    )


def build_json_fallback_prompt(
    user_message: str,
    state_summary: dict,
    locations: dict,
    missions: dict | None = None,
    identity: dict | None = None,
    chargers: list | None = None,
) -> str:
    return (
        "Respond with JSON only.\n"
        "Schema:\n"
        '{"response":"short user-facing reply","action":{"tool":"tool_name","arguments":{}}}\n'
        'If no tool is needed, set "action" to null.\n'
        "Allowed tools: get_robot_status, list_locations, move_to_location, cancel_current_task, return_to_charger, emergency_stop, release_emergency_stop_confirmed, get_mission_status, list_missions.\n"
        "Only send the robot to a charger whose is_mine is true; use return_to_charger to recharge.\n"
        f"{build_runtime_context(state_summary, locations, missions, identity, chargers)}"
        f"User message: {user_message}\n"
    )
