from pathlib import Path

from app.config import Settings
from app.db import create_session_factory, init_db
from app.robot.locations import LocationRegistry
from app.robot.safety import SafetyValidator
from app.robot.state_manager import StateManager
from app.robot.task_manager import TaskManager
from app.water.client import WaterRobotClient


def build_manager(tmp_path: Path) -> TaskManager:
    db_url = f"sqlite:///{tmp_path / 'task.db'}"
    settings = Settings(database_url=db_url, water_dry_run=True)
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    client = WaterRobotClient(settings)
    state_manager = StateManager(client)
    registry = LocationRegistry(session_factory, client)
    registry.sync_markers()
    safety = SafetyValidator(settings)
    return TaskManager(session_factory, client, state_manager, registry, safety)


def test_move_task_lifecycle_in_dry_run(tmp_path: Path):
    manager = build_manager(tmp_path)
    task = manager.create_move_marker_task("front_desk")
    assert task["status"] == "running"
    manager.poll_active_tasks()
    manager.poll_active_tasks()
    task_after = manager.get_task(task["task_id"])
    assert task_after["status"] in {"running", "succeeded"}
    manager.poll_active_tasks()
    assert manager.get_task(task["task_id"])["status"] == "succeeded"


def test_cancel_and_estop_tasks(tmp_path: Path):
    manager = build_manager(tmp_path)
    cancel_task = manager.cancel_current_move()
    estop_task = manager.emergency_stop()
    assert cancel_task["status"] == "succeeded"
    assert estop_task["status"] == "succeeded"
