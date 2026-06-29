import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def build_test_client(tmp_path: Path, **overrides) -> TestClient:
    params = dict(
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        water_dry_run=True,
        llm_enabled=True,
        llm_dry_run=True,
    )
    params.update(overrides)
    return TestClient(create_app(Settings(**params)))


def test_health_and_task_routes(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        assert client.get("/health").json()["ok"] is True
        task = client.post("/tasks/move-marker", json={"marker_name": "front_desk"}).json()
        assert task["task_type"] == "move_marker"
        assert client.get(f"/tasks/{task['task_id']}").status_code == 200


def test_agent_chat_json_fallback(tmp_path: Path):
    # llm_dry_run=False so run_chat takes the live tool path; the mocks then
    # force the tool call to fail and exercise the JSON fallback.
    with build_test_client(tmp_path, llm_dry_run=False) as client:
        planner = client.app.state.services.agent_planner
        planner.llm_client.chat_with_tools = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("unsupported"))  # type: ignore[method-assign]
        planner.llm_client.json_plan = lambda prompt: {  # type: ignore[method-assign]
            "response": "Sending the robot to the front desk.",
            "action": {"tool": "move_to_location", "arguments": {"location_name": "front_desk"}},
        }
        response = client.post("/agent/chat", json={"message": "Go to the front desk"})
        payload = response.json()
        assert response.status_code == 200
        assert payload["created_task_ids"]


def test_agent_chat_creates_mission_via_tool(tmp_path: Path):
    # Live tool path: the model calls the create_mission tool, then replies.
    with build_test_client(tmp_path, llm_dry_run=False) as client:
        planner = client.app.state.services.agent_planner
        calls = {"n": 0}

        def fake_chat(messages, tools):
            calls["n"] += 1
            if calls["n"] == 1:
                args = json.dumps({
                    "steps": [
                        {"step_type": "move_marker", "marker_name": "front_desk"},
                        {"step_type": "return_to_charger"},
                    ],
                    "mission_name": "demo",
                })
                return {"choices": [{"message": {"content": "", "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "create_mission", "arguments": args}},
                ]}}]}
            return {"choices": [{"message": {"content": "Mission created.", "tool_calls": []}}]}

        planner.llm_client.chat_with_tools = fake_chat  # type: ignore[method-assign]
        response = client.post("/agent/chat", json={"message": "go to front desk then return to charger"})
        payload = response.json()
        assert response.status_code == 200
        assert payload["created_mission_ids"]


def test_run_chat_injects_history_and_summary(tmp_path: Path):
    with build_test_client(tmp_path, llm_dry_run=False) as client:
        planner = client.app.state.services.agent_planner
        captured: dict = {}

        def fake_chat(messages, tools):
            captured["messages"] = messages
            return {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}

        planner.llm_client.chat_with_tools = fake_chat  # type: ignore[method-assign]
        history = [
            {"role": "user", "content": "go to kitchen"},
            {"role": "assistant", "content": "done"},
        ]
        planner.run_chat("now come back", history=history, summary="Earlier: robot was idle.")
        messages = captured["messages"]
        assert any("Earlier: robot was idle." in (m.get("content") or "") for m in messages)
        assert {"role": "user", "content": "go to kitchen"} in messages
        assert messages[-1] == {"role": "user", "content": "now come back"}


def test_agent_chat_creates_mission_for_multistep_request(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        response = client.post("/agent/chat", json={"message": "Go to front desk, then wait 1 second, then return to charger"})
        payload = response.json()
        assert response.status_code == 200
        assert payload["created_mission_ids"]
        mission_id = payload["created_mission_ids"][0]
        mission = client.get(f"/missions/{mission_id}").json()
        assert len(mission["steps"]) >= 2


def test_mission_routes(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        response = client.post(
            "/missions",
            json={
                "user_request": "Go to the front desk then return to charger",
                "steps": [
                    {"step_type": "move_marker", "marker_name": "front_desk"},
                    {"step_type": "return_to_charger"},
                ],
                "auto_replan": True,
            },
        )
        mission = response.json()
        assert response.status_code == 200
        assert mission["status"] == "created"
        assert client.get(f"/missions/{mission['mission_id']}/context").status_code == 200
