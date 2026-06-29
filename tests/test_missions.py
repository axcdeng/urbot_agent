from pathlib import Path

from app.agent.llm_client import LLMClient
from app.agent.mission_planner import MissionPlanner
from app.config import Settings
from app.db import create_session_factory, init_db
from app.robot.locations import LocationRegistry
from app.robot.mission_manager import MissionManager
from app.robot.safety import SafetyValidator
from app.robot.state_manager import StateManager
from app.robot.task_manager import TaskManager
from app.water.client import WaterRobotClient


def build_mission_manager(tmp_path: Path) -> tuple[MissionManager, TaskManager, MissionPlanner]:
    db_url = f"sqlite:///{tmp_path / 'mission.db'}"
    settings = Settings(database_url=db_url, water_dry_run=True, llm_enabled=False, llm_dry_run=True)
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    client = WaterRobotClient(settings)
    state_manager = StateManager(client)
    registry = LocationRegistry(session_factory, client)
    registry.sync_markers()
    safety = SafetyValidator(settings)
    task_manager = TaskManager(session_factory, client, state_manager, registry, safety)
    llm_client = LLMClient(settings)
    mission_planner = MissionPlanner(llm_client, registry, state_manager, settings)
    mission_manager = MissionManager(session_factory, task_manager, state_manager, registry, settings, mission_planner=mission_planner)
    return mission_manager, task_manager, mission_planner


def test_mission_runs_steps_sequentially(tmp_path: Path):
    mission_manager, task_manager, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Go to the front desk then wait then return to charger",
        steps=[
            {"step_type": "move_marker", "marker_name": "front_desk"},
            {"step_type": "wait", "wait_seconds": 1},
            {"step_type": "return_to_charger"},
        ],
        auto_replan=False,
    )
    mission_id = mission["mission_id"]

    for _ in range(10):
        task_manager.poll_active_tasks()
        mission_manager.poll_missions()

    result = mission_manager.get_mission(mission_id)
    assert result["steps"][0]["status"] in {"running", "succeeded"}

    import time

    time.sleep(1.1)
    for _ in range(12):
        task_manager.poll_active_tasks()
        mission_manager.poll_missions()

    result = mission_manager.get_mission(mission_id)
    assert result["status"] == "succeeded"
    assert [step["status"] for step in result["steps"]] == ["succeeded", "succeeded", "succeeded"]


def test_soft_stop_finishes_current_step_then_stops(tmp_path: Path):
    mission_manager, task_manager, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Visit three places",
        steps=[
            {"step_type": "move_marker", "marker_name": "front_desk"},
            {"step_type": "move_marker", "marker_name": "Kitchen"},
            {"step_type": "move_marker", "marker_name": "Meetingroom"},
        ],
        auto_replan=False,
    )
    mission_id = mission["mission_id"]

    # Dispatch step 0 -> running.
    mission_manager.poll_missions()
    assert mission_manager.get_mission(mission_id)["steps"][0]["status"] == "running"

    # Soft stop: the running step is left alone; the two pending steps are canceled.
    report = mission_manager.soft_stop_mission(mission_id)
    assert report["canceled_pending_steps"] == 2
    assert report["finishing_current_step"] is True

    # Let the in-progress move finish; the mission then stops (no more steps run).
    for _ in range(6):
        task_manager.poll_active_tasks()
        mission_manager.poll_missions()

    result = mission_manager.get_mission(mission_id)
    assert result["status"] == "canceled"  # soft-stopped, not "succeeded"
    statuses = [step["status"] for step in result["steps"]]
    assert statuses[0] == "succeeded"  # current step ran to completion
    assert statuses[1] == "canceled" and statuses[2] == "canceled"


def test_mission_replan_uses_compact_context(tmp_path: Path):
    mission_manager, task_manager, mission_planner = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Go to the front desk then return to charger",
        steps=[{"step_type": "move_marker", "marker_name": "front_desk"}],
        auto_replan=True,
    )
    mission_id = mission["mission_id"]

    mission_manager.poll_missions()
    task_manager.get_task = lambda task_id: {  # type: ignore[method-assign]
        "task_id": task_id,
        "task_type": "move_marker",
        "status": "failed",
        "requested_target": "front_desk",
        "robot_task_id": "robot-1",
        "mission_id": mission_id,
        "mission_step_id": mission["steps"][0]["step_id"],
        "error_message": "Path blocked",
        "created_at": None,
        "updated_at": None,
        "completed_at": None,
    }
    mission_planner.replan_steps = lambda context: {  # type: ignore[method-assign]
        "response": "Use the kitchen route first.",
        "steps": [
            {"step_type": "move_marker", "marker_name": "Kitchen"},
            {"step_type": "return_to_charger"},
        ],
    }

    mission_manager.poll_missions()
    replanned = mission_manager.get_mission(mission_id)
    assert replanned["replan_count"] == 1
    assert replanned["status"] == "running"
    assert "Use the kitchen route first." in (replanned["error_message"] or "")


def test_compact_summary_rolls_up_older_completed_steps(tmp_path: Path):
    mission_manager, task_manager, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Run many fast steps",
        steps=[{"step_type": "cancel_move", "description": f"Cancel step {idx}"} for idx in range(8)],
        auto_replan=False,
    )
    for _ in range(12):
        task_manager.poll_active_tasks()
        mission_manager.poll_missions()

    result = mission_manager.get_mission(mission["mission_id"])
    assert result["status"] == "succeeded"
    assert "Earlier completed steps" in (result["context_summary"] or "")
