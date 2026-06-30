import json
from pathlib import Path

from app.config import Settings
from app.main import build_services
from app.tui import AgentConsole


def _scripted_llm(console: AgentConsole, responses: list[dict]) -> None:
    """Make the agent's tool-calling LLM return a fixed sequence of responses,
    and stub the plain-text completion (titles/summaries) so nothing hits the
    network."""
    calls = {"i": 0}

    def fake_chat_with_tools(messages, tools):
        response = responses[calls["i"]]
        calls["i"] += 1
        return response

    console.services.llm_client.chat_with_tools = fake_chat_with_tools  # type: ignore[method-assign]
    console.services.llm_client.complete = lambda *a, **k: ""  # type: ignore[method-assign]


def _assistant_turn(content: str, tool_calls: list[tuple] | None = None) -> dict:
    calls = []
    for i, (name, args) in enumerate(tool_calls or []):
        calls.append({"id": f"c{i}", "function": {"name": name, "arguments": json.dumps(args)}})
    return {"choices": [{"message": {"content": content, "tool_calls": calls}}]}


def build_console(tmp_path: Path, **overrides) -> AgentConsole:
    kwargs = dict(
        database_url=f"sqlite:///{tmp_path / 'tui.db'}",
        water_dry_run=True,
        llm_enabled=False,
        llm_dry_run=True,
    )
    kwargs.update(overrides)
    return AgentConsole(build_services(Settings(**kwargs)))


def test_dryrun_toggle(tmp_path: Path):
    console = build_console(tmp_path)
    assert console.dry_run is True
    assert "ON" in console.run_command("/dryrun").lines[0]

    result = console.run_command("/dryrun off")
    assert console.dry_run is False
    assert "OFF" in result.lines[0]
    assert console.live_target in result.lines[0]

    console.run_command("/dryrun on")
    assert console.dry_run is True


def test_markers_command_lists_real_waypoints(tmp_path: Path):
    console = build_console(tmp_path)
    text = " ".join(console.run_command("/markers").lines)
    assert "front_desk" in text
    assert "Meetingroom" in text
    assert "charger -> charge_point_1F_1" in text


def test_ip_command_switches_robot_address(tmp_path: Path):
    console = build_console(tmp_path)
    assert console.robot_addr in console.run_command("/ip").lines[0]

    console.run_command("/ip 10.9.9.9")
    assert console.services.settings.water_robot_host == "10.9.9.9"
    # client builds URLs from the live setting -> redirected immediately
    assert console.services.client.get_transport_url("/api/move").startswith("http://10.9.9.9:9001")

    console.run_command("/ip 10.0.0.5:8000")
    assert console.services.settings.water_robot_host == "10.0.0.5"
    assert console.services.settings.water_http_port == 8000
    assert console.services.client.get_transport_url("/api/move").startswith("http://10.0.0.5:8000")

    assert "invalid port" in console.run_command("/ip 10.0.0.5:notaport").lines[0]
    assert console.services.settings.water_http_port == 8000


def test_ip_profile_switches_address_and_map(tmp_path: Path):
    console = build_console(tmp_path)

    out = console.run_command("/ip secondary")
    assert "secondary" in out.lines[0]
    assert console.services.settings.water_robot_host == "10.1.16.160"
    assert console.services.settings.water_http_port == 9001
    assert console.current_profile() == "secondary"
    # map is loaded and resolvable; dry simulation uses it too
    assert console.services.client.get_transport_url("/api/move").startswith("http://10.1.16.160:9001")
    assert console.services.location_registry.resolve_location("Meetingroom").marker_name == "Meetingroom"
    assert "front_desk" in console.services.client.dry_markers

    console.run_command("/ip primary")
    assert console.services.settings.water_robot_host == "10.1.17.225"
    assert console.current_profile() == "primary"


def test_meta_commands(tmp_path: Path):
    console = build_console(tmp_path)
    assert console.run_command("/help").lines
    assert console.run_command("/quit").quit is True
    assert console.run_command("/clear").clear is True
    assert "unknown command" in console.run_command("/bogus").lines[0]


def test_soft_estop_and_release(tmp_path: Path):
    console = build_console(tmp_path)
    console.soft_estop()
    assert console.status()["estop_state"] is True
    console.release_estop()
    assert console.status()["estop_state"] is False


def test_panic_cancels_active_move(tmp_path: Path):
    console = build_console(tmp_path)
    console.soft_estop()
    # cancel_active_move must succeed even after an e-stop (dry-run path).
    assert console.cancel_active_move()["status"] == "succeeded"


def test_chat_returns_structured_result_when_llm_disabled(tmp_path: Path):
    console = build_console(tmp_path)
    result = console.chat("hello there")
    assert "assistant_response" in result
    assert "tool_calls" in result
    # metrics are attached for the turn (zero calls when the LLM is disabled)
    assert "metrics" in result
    assert result["metrics"].calls == 0


def test_chat_with_then_but_no_movement_does_not_crash(tmp_path: Path):
    # "and then" trips the mission heuristic, but there is no movement step;
    # this must answer normally, not raise "A mission requires at least one step."
    console = build_console(tmp_path)
    result = console.chat("Hey, what's your battery and then list all the locations you can go to")
    assert "assistant_response" in result
    assert not result.get("created_mission_ids")


def test_chat_has_memory_within_session(tmp_path: Path):
    console = build_console(tmp_path)
    console.chat("first message")
    _, history = console.convo.build_history(console.current_session_id)
    assert len(history) == 2  # user + assistant
    console.chat("second message")
    _, history = console.convo.build_history(console.current_session_id)
    assert len(history) == 4


def test_new_chats_and_open_commands(tmp_path: Path):
    console = build_console(tmp_path)
    console.chat("go to the kitchen")
    first = console.current_session_id

    res = console.run_command("/new")
    assert res.clear is True
    assert console.current_session_id != first
    console.chat("what is your battery")

    chats = console.run_command("/chats").lines
    assert len(chats) >= 3  # header + 2 chats

    sessions = console.list_chats()
    idx = next(i for i, s in enumerate(sessions, 1) if s["session_id"] == first)
    out = console.run_command(f"/open {idx}")
    assert out.clear is True
    assert console.current_session_id == first
    assert any("kitchen" in line.lower() for line in out.lines)


def test_soft_stop_command(tmp_path: Path):
    console = build_console(tmp_path)
    # Nothing running -> nothing to gracefully stop.
    assert "no active mission" in console.run_command("/stop").lines[0]
    # Build a multi-step mission via the offline heuristic, dispatch step 0, stop.
    result = console.chat("go to front desk, then go to Kitchen, then go to Meetingroom")
    assert result.get("created_mission_ids")
    console.poll()  # dispatches the first step
    out = console.run_command("/stop")
    assert "soft stop" in out.lines[0].lower()


def test_compact_current_offline(tmp_path: Path):
    console = build_console(tmp_path, compact_keep_recent_turns=1)
    for i in range(4):
        console.chat(f"message {i}")
    report = console.compact_current()
    assert report["compacted"] is True


def test_chat_streams_events_and_status_update_is_narration_not_a_tool(tmp_path: Path):
    console = build_console(tmp_path, llm_enabled=True, llm_dry_run=False)
    _scripted_llm(console, [
        _assistant_turn("", [
            ("status_update", {"message": "On it."}),
            ("get_robot_status", {}),
        ]),
        _assistant_turn("Battery looks fine.", []),
    ])

    events: list[dict] = []
    result = console.chat("how's the battery", on_event=events.append)

    # status_update -> narration (rendered live); the real tool -> call + result.
    assert [e["type"] for e in events] == ["narration", "tool_call", "tool_result"]
    assert events[0]["text"] == "On it."
    assert events[1]["name"] == "get_robot_status"
    assert result["assistant_response"] == "Battery looks fine."
    assert result["streamed"] is True
    # status_update performs no action, so it is NOT recorded as a tool call.
    assert [c["name"] for c in result["tool_calls"]] == ["get_robot_status"]


def test_status_only_rounds_do_not_exhaust_the_action_budget(tmp_path: Path):
    console = build_console(tmp_path, llm_enabled=True, llm_dry_run=False)
    # Five pure-narration rounds must not burn the 4-action budget; the real
    # action and final reply still land.
    status_round = _assistant_turn("", [("status_update", {"message": "working"})])
    _scripted_llm(console, [status_round] * 5 + [
        _assistant_turn("", [("get_robot_status", {})]),
        _assistant_turn("done", []),
    ])

    result = console.chat("status please")
    assert result["assistant_response"] == "done"
    assert [c["name"] for c in result["tool_calls"]] == ["get_robot_status"]
