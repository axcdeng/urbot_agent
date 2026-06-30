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
def test_poll_advances_multiple_steps_in_one_pass(tmp_path: Path):
    """A single poll pass should drive all ready steps to completion.

    Previously the poller advanced exactly one step transition per pass, so a
    chain of instant steps (or a completed step's successor) waited a full poll
    interval each. That stacked poll-interval latency between every step.
    """
    mission_manager, _, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Run three fast steps",
        steps=[{"step_type": "cancel_move", "description": f"Cancel {idx}"} for idx in range(3)],
        auto_replan=False,
    )

    mission_manager.poll_missions()

    result = mission_manager.get_mission(mission["mission_id"])
    assert result["status"] == "succeeded"
    assert [step["status"] for step in result["steps"]] == ["succeeded", "succeeded", "succeeded"]


def test_wait_completion_dispatches_next_step_in_same_pass(tmp_path: Path):
    """When a wait's timer elapses, the same poll pass should also start the next step.

    The wait timer itself is wall-clock correct; the bug was that completing the
    wait and dispatching the following step happened in separate poll passes,
    adding a poll interval of dead time after every wait.
    """
    import time

    mission_manager, _, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Wait then cancel",
        steps=[
            {"step_type": "wait", "wait_seconds": 1},
            {"step_type": "cancel_move"},
        ],
        auto_replan=False,
    )

    # First pass arms the wait (PENDING -> WAITING) and, since it has not
    # elapsed, blocks there.
    mission_manager.poll_missions()
    assert mission_manager.get_mission(mission["mission_id"])["steps"][0]["status"] == "waiting"

    time.sleep(1.1)

    # A single pass after the timer elapses must complete the wait AND run the
    # next step, finishing the mission.
    mission_manager.poll_missions()

    result = mission_manager.get_mission(mission["mission_id"])
    assert result["status"] == "succeeded"
    assert [step["status"] for step in result["steps"]] == ["succeeded", "succeeded"]


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


def test_claim_step_is_exclusive(tmp_path: Path):
    """Only the first caller may claim a PENDING step; a second caller is rejected."""
    mission_manager, _, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Go to the front desk",
        steps=[{"step_type": "move_marker", "marker_name": "front_desk"}],
        auto_replan=False,
    )
    step_id = mission["steps"][0]["step_id"]

    assert mission_manager._claim_step(step_id) is True
    # Already claimed (now RUNNING) -> the second claim must fail.
    assert mission_manager._claim_step(step_id) is False


def test_claim_step_is_exclusive_under_concurrency(tmp_path: Path):
    """When many pollers race to claim the same PENDING step, exactly one wins.

    This is the core guarantee that stops two pollers from double-dispatching a
    step and sending duplicate move commands to the robot. A barrier releases all
    threads at once so they genuinely contend on the conditional UPDATE.
    """
    import threading

    mission_manager, _, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Go to the front desk",
        steps=[{"step_type": "move_marker", "marker_name": "front_desk"}],
        auto_replan=False,
    )
    step_id = mission["steps"][0]["step_id"]

    worker_count = 8
    barrier = threading.Barrier(worker_count)
    results: list[bool] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        won = mission_manager._claim_step(step_id)
        with lock:
            results.append(won)

    threads = [threading.Thread(target=worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert results.count(True) == 1, f"expected exactly one winner, got {results}"


def test_poll_does_not_redispatch_a_claimed_step(tmp_path: Path):
    """A step another poller has already claimed (RUNNING) must not be dispatched again."""
    mission_manager, task_manager, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Go to the front desk",
        steps=[{"step_type": "move_marker", "marker_name": "front_desk"}],
        auto_replan=False,
    )
    mission_id = mission["mission_id"]
    step_id = mission["steps"][0]["step_id"]

    # Simulate another poller having just claimed the step.
    assert mission_manager._claim_step(step_id) is True

    mission_manager.poll_missions()

    tasks = [t for t in task_manager.list_tasks() if t["mission_id"] == mission_id]
    assert tasks == [], f"claimed step was re-dispatched: {[t['id'] for t in tasks]}"


def _backdate_step(mission_manager, step_id: str, seconds: int) -> None:
    from datetime import datetime, timedelta, timezone

    from app.models import MissionStepRecord

    session = mission_manager.session_factory()
    try:
        step = session.get(MissionStepRecord, step_id)
        step.updated_at = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        session.commit()
    finally:
        session.close()


def test_orphaned_running_step_is_requeued(tmp_path: Path):
    """A step left RUNNING with no task_id by a crashed poller is recovered and runs."""
    mission_manager, task_manager, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Go to the front desk",
        steps=[{"step_type": "move_marker", "marker_name": "front_desk"}],
        auto_replan=False,
    )
    mission_id = mission["mission_id"]
    step_id = mission["steps"][0]["step_id"]

    # Claim leaves it RUNNING with no task_id (as if the dispatcher died here).
    assert mission_manager._claim_step(step_id) is True
    _backdate_step(mission_manager, step_id, seconds=999)

    # First poll recovers (re-queues) the orphan; subsequent polls dispatch it.
    for _ in range(4):
        task_manager.poll_active_tasks()
        mission_manager.poll_missions()

    tasks = [t for t in task_manager.list_tasks() if t["mission_id"] == mission_id]
    assert len(tasks) == 1, f"expected one recovered dispatch, got {len(tasks)}"
    assert mission_manager.get_mission(mission_id)["steps"][0]["status"] in {"running", "succeeded"}


def test_recently_claimed_step_is_not_treated_as_orphan(tmp_path: Path):
    """A freshly-claimed (in-flight) step must NOT be re-queued by the orphan recovery."""
    mission_manager, task_manager, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Go to the front desk",
        steps=[{"step_type": "move_marker", "marker_name": "front_desk"}],
        auto_replan=False,
    )
    mission_id = mission["mission_id"]
    step_id = mission["steps"][0]["step_id"]

    assert mission_manager._claim_step(step_id) is True  # recent, no task yet

    mission_manager.poll_missions()

    # No task was created, and it must remain claimed (not bounced back to pending).
    tasks = [t for t in task_manager.list_tasks() if t["mission_id"] == mission_id]
    assert tasks == []
    assert mission_manager.get_mission(mission_id)["steps"][0]["status"] == "running"


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


def test_drain_finalized_reports_completion_once(tmp_path: Path):
    """A mission that finishes is reported by drain_finalized exactly once."""
    mission_manager, task_manager, _ = build_mission_manager(tmp_path)
    mission = mission_manager.create_mission(
        user_request="Run two fast steps",
        steps=[{"step_type": "cancel_move", "description": f"Cancel {idx}"} for idx in range(2)],
        auto_replan=False,
    )
    mission_id = mission["mission_id"]

    # Nothing has finished yet.
    assert mission_manager.drain_finalized() == []

    for _ in range(6):
        task_manager.poll_active_tasks()
        mission_manager.poll_missions()
    assert mission_manager.get_mission(mission_id)["status"] == "succeeded"

    finished = mission_manager.drain_finalized()
    assert [m["mission_id"] for m in finished] == [mission_id]
    assert finished[0]["status"] == "succeeded"
    # Draining is one-shot: a second drain (and further polls) reports nothing.
    assert mission_manager.drain_finalized() == []
    mission_manager.poll_missions()
    assert mission_manager.drain_finalized() == []
