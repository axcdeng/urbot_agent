from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def build_test_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        water_dry_run=True,
        llm_enabled=True,
        llm_dry_run=True,
    )
    return TestClient(create_app(settings))


def test_health_and_task_routes(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        assert client.get("/health").json()["ok"] is True
        task = client.post("/tasks/move-marker", json={"marker_name": "room_205"}).json()
        assert task["task_type"] == "move_marker"
        assert client.get(f"/tasks/{task['task_id']}").status_code == 200


def test_agent_chat_json_fallback(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        planner = client.app.state.services.agent_planner
        planner.llm_client.chat_with_tools = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("unsupported"))  # type: ignore[method-assign]
        planner.llm_client.json_plan = lambda prompt: {  # type: ignore[method-assign]
            "response": "Sending the robot to room 205.",
            "action": {"tool": "move_to_location", "arguments": {"location_name": "room_205"}},
        }
        response = client.post("/agent/chat", json={"message": "Go to room 205"})
        payload = response.json()
        assert response.status_code == 200
        assert payload["created_task_ids"]


def test_agent_chat_creates_mission_for_multistep_request(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        response = client.post("/agent/chat", json={"message": "Go to room 205, then wait 1 second, then return to charger"})
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
                "user_request": "Go to room 205 then return to charger",
                "steps": [
                    {"step_type": "move_marker", "marker_name": "room_205"},
                    {"step_type": "return_to_charger"},
                ],
                "auto_replan": True,
            },
        )
        mission = response.json()
        assert response.status_code == 200
        assert mission["status"] == "created"
        assert client.get(f"/missions/{mission['mission_id']}/context").status_code == 200
