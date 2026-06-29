from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models import MissionRecord, MissionStatus, MissionStepRecord, MissionStepStatus, MissionStepType, TaskStatus
from app.robot.locations import LocationRegistry
from app.robot.state_manager import StateManager
from app.robot.task_manager import TaskManager


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


FINAL_MISSION_STATUSES = {MissionStatus.SUCCEEDED.value, MissionStatus.FAILED.value, MissionStatus.CANCELED.value}
FINAL_STEP_STATUSES = {MissionStepStatus.SUCCEEDED.value, MissionStepStatus.FAILED.value, MissionStepStatus.CANCELED.value}


@dataclass
class MissionResult:
    mission: MissionRecord
    steps: list[MissionStepRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission.id,
            "name": self.mission.name,
            "user_request": self.mission.user_request,
            "status": self.mission.status,
            "current_step_index": self.mission.current_step_index,
            "auto_replan": self.mission.auto_replan,
            "replan_count": self.mission.replan_count,
            "error_message": self.mission.error_message,
            "context_summary": self.mission.context_summary,
            "compact_state": self.mission.compact_state,
            "created_at": self.mission.created_at.isoformat() if self.mission.created_at else None,
            "updated_at": self.mission.updated_at.isoformat() if self.mission.updated_at else None,
            "started_at": self.mission.started_at.isoformat() if self.mission.started_at else None,
            "completed_at": self.mission.completed_at.isoformat() if self.mission.completed_at else None,
            "steps": [
                {
                    "step_id": step.id,
                    "step_index": step.step_index,
                    "step_type": step.step_type,
                    "status": step.status,
                    "description": step.description,
                    "payload": step.payload,
                    "task_id": step.task_id,
                    "error_message": step.error_message,
                    "compact_result": step.compact_result,
                    "created_at": step.created_at.isoformat() if step.created_at else None,
                    "updated_at": step.updated_at.isoformat() if step.updated_at else None,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                }
                for step in self.steps
            ],
        }


class MissionManager:
    def __init__(
        self,
        session_factory: sessionmaker,
        task_manager: TaskManager,
        state_manager: StateManager,
        location_registry: LocationRegistry,
        settings: Settings,
        mission_planner=None,
    ):
        self.session_factory = session_factory
        self.task_manager = task_manager
        self.state_manager = state_manager
        self.location_registry = location_registry
        self.settings = settings
        self.mission_planner = mission_planner
        self._polling_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        # A step claimed for dispatch but never linked to a task is "in flight"
        # only briefly (bounded by the move HTTP timeout). Past this grace period
        # we treat it as orphaned by a crashed poller and recover it. Kept well
        # above the HTTP timeout so we never disturb a dispatch still in progress.
        self._orphan_grace_seconds = max(30.0, settings.water_timeout_seconds * 2)

    def _step_description(self, step: dict[str, Any]) -> str:
        step_type = step["step_type"]
        if step_type == MissionStepType.MOVE_MARKER.value:
            return f"Move to {step['marker_name']}"
        if step_type == MissionStepType.MOVE_COORDINATE.value:
            return f"Move to coordinate {step['x']},{step['y']},{step['theta']}"
        if step_type == MissionStepType.WAIT.value:
            return f"Wait {step['wait_seconds']} seconds"
        if step_type == MissionStepType.RETURN_TO_CHARGER.value:
            return "Return to charger"
        if step_type == MissionStepType.CANCEL_MOVE.value:
            return "Cancel current move"
        if step_type == MissionStepType.EMERGENCY_STOP.value:
            return "Emergency stop"
        return step_type

    def _validate_step(self, step: dict[str, Any]) -> dict[str, Any]:
        step_type = step["step_type"]
        if step_type == MissionStepType.MOVE_MARKER.value:
            resolved = self.location_registry.resolve_location(step["marker_name"])
            if resolved.marker_name is None:
                raise ValueError(f"Unknown mission location '{step['marker_name']}'.")
        if step_type == MissionStepType.WAIT.value and int(step.get("wait_seconds", 0)) <= 0:
            raise ValueError("Wait steps require wait_seconds > 0.")
        return step

    def create_mission(self, *, user_request: str, steps: list[dict[str, Any]], name: str | None = None, auto_replan: bool = False) -> dict[str, Any]:
        if not steps:
            raise ValueError("A mission requires at least one step.")
        validated_steps = [self._validate_step(dict(step)) for step in steps]
        session = self.session_factory()
        try:
            mission = MissionRecord(
                name=name,
                user_request=user_request,
                status=MissionStatus.CREATED.value,
                auto_replan=auto_replan,
                current_step_index=0,
            )
            session.add(mission)
            session.flush()
            records: list[MissionStepRecord] = []
            for index, step in enumerate(validated_steps):
                records.append(
                    MissionStepRecord(
                        mission_id=mission.id,
                        step_index=index,
                        step_type=step["step_type"],
                        status=MissionStepStatus.PENDING.value,
                        description=step.get("description") or self._step_description(step),
                        payload=step,
                    )
                )
            session.add_all(records)
            self._refresh_summary(mission, records)
            session.commit()
            return MissionResult(mission, records).to_dict()
        finally:
            session.close()

    def list_missions(self) -> list[dict[str, Any]]:
        session = self.session_factory()
        try:
            missions = session.scalars(select(MissionRecord).order_by(MissionRecord.created_at.desc())).all()
            results = []
            for mission in missions:
                steps = session.scalars(select(MissionStepRecord).where(MissionStepRecord.mission_id == mission.id).order_by(MissionStepRecord.step_index)).all()
                results.append(MissionResult(mission, list(steps)).to_dict())
            return results
        finally:
            session.close()

    def get_mission(self, mission_id: str) -> dict[str, Any]:
        session = self.session_factory()
        try:
            mission = session.get(MissionRecord, mission_id)
            if mission is None:
                raise ValueError(f"Mission '{mission_id}' not found.")
            steps = session.scalars(select(MissionStepRecord).where(MissionStepRecord.mission_id == mission.id).order_by(MissionStepRecord.step_index)).all()
            return MissionResult(mission, list(steps)).to_dict()
        finally:
            session.close()

    def cancel_mission(self, mission_id: str) -> dict[str, Any]:
        session = self.session_factory()
        try:
            mission = session.get(MissionRecord, mission_id)
            if mission is None:
                raise ValueError(f"Mission '{mission_id}' not found.")
            steps = list(session.scalars(select(MissionStepRecord).where(MissionStepRecord.mission_id == mission_id).order_by(MissionStepRecord.step_index)).all())
            mission.status = MissionStatus.CANCELED.value
            mission.completed_at = utcnow()
            mission.error_message = "Mission canceled by user."
            for step in steps:
                if step.status not in FINAL_STEP_STATUSES:
                    step.status = MissionStepStatus.CANCELED.value
                    step.completed_at = utcnow()
            self._refresh_summary(mission, steps)
            session.commit()
            return MissionResult(mission, steps).to_dict()
        finally:
            session.close()

    def get_mission_context(self, mission_id: str) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        return {
            "mission_id": mission["mission_id"],
            "status": mission["status"],
            "current_step_index": mission["current_step_index"],
            "user_request": mission["user_request"],
            "replan_count": mission["replan_count"],
            "context_summary": mission["context_summary"],
            "compact_state": mission["compact_state"],
            "steps": mission["steps"],
        }

    def _refresh_summary(self, mission: MissionRecord, steps: list[MissionStepRecord]) -> None:
        recent_count = self.settings.agent_max_recent_events
        completed_steps = [step for step in steps if step.status == MissionStepStatus.SUCCEEDED.value]
        recent_completed = completed_steps[-recent_count:]
        current_step = next((step for step in steps if step.status not in FINAL_STEP_STATUSES), None)
        next_steps = [step.description for step in steps if step.status == MissionStepStatus.PENDING.value][:3]
        older_count = max(0, len(completed_steps) - len(recent_completed))
        summary_lines = [
            f"Mission status: {mission.status}",
            f"Goal: {mission.user_request}",
            f"Replans used: {mission.replan_count}/{self.settings.mission_max_replans}",
        ]
        if older_count:
            summary_lines.append(f"Earlier completed steps: {older_count}")
        if recent_completed:
            summary_lines.append("Recent completed: " + "; ".join(step.description or step.step_type for step in recent_completed))
        if current_step is not None:
            summary_lines.append(f"Current step: {current_step.step_index + 1}. {current_step.description} [{current_step.status}]")
        if next_steps:
            summary_lines.append("Next steps: " + "; ".join(next_steps))
        if mission.error_message:
            summary_lines.append(f"Last error: {mission.error_message}")
        mission.context_summary = "\n".join(summary_lines)
        mission.compact_state = {
            "current_step_index": mission.current_step_index,
            "status": mission.status,
            "recent_completed_steps": [step.description for step in recent_completed],
            "pending_steps": next_steps,
            "error_message": mission.error_message,
        }

    def _claim_step(self, step_id: str) -> bool:
        """Atomically transition a PENDING step to RUNNING so only one poller dispatches it.

        Multiple pollers (e.g. more than one running process, all pointed at the
        same database) can each see a step as PENDING and dispatch it, sending
        duplicate move commands to the robot — which the robot then rejects with
        "Robot is already moving", failing the mission. This conditional UPDATE is
        serialized by the database, so exactly one caller observes rowcount == 1
        and is allowed to dispatch; everyone else gets False and backs off.

        Uses its own short-lived session and commits immediately, so the claim is
        durable before the slow move dispatch runs.
        """
        session = self.session_factory()
        try:
            result = session.execute(
                update(MissionStepRecord)
                .where(MissionStepRecord.id == step_id)
                .where(MissionStepRecord.status == MissionStepStatus.PENDING.value)
                .values(status=MissionStepStatus.RUNNING.value, updated_at=utcnow())
            )
            session.commit()
            return result.rowcount == 1
        finally:
            session.close()

    def _recover_orphaned_step(self, step: MissionStepRecord) -> None:
        """Recover a step stuck RUNNING with no task_id (its dispatcher crashed).

        A step holds this state only between being claimed and its task being
        recorded — normally within one poll pass. If it persists, the poller that
        claimed it died mid-dispatch. If a task was actually created we re-link to
        it; otherwise, once the claim is stale beyond the grace period (so we never
        clobber a dispatch still running in another poller), we re-queue the step.
        """
        task = self.task_manager.find_latest_task_for_step(step.id)
        if task is not None:
            step.task_id = task["task_id"]
            step.updated_at = utcnow()
            return
        last_update = ensure_utc(step.updated_at) if step.updated_at else None
        if last_update is None or (utcnow() - last_update).total_seconds() >= self._orphan_grace_seconds:
            step.status = MissionStepStatus.PENDING.value
            step.started_at = None
            step.updated_at = utcnow()

    def _mark_step_complete(self, mission: MissionRecord, step: MissionStepRecord, status: MissionStepStatus, *, result: dict[str, Any] | None = None, error_message: str | None = None) -> None:
        step.status = status.value
        step.compact_result = result
        step.error_message = error_message
        step.updated_at = utcnow()
        step.completed_at = utcnow()
        mission.current_step_index = step.step_index + 1

    def _dispatch_step(self, mission: MissionRecord, step: MissionStepRecord) -> None:
        payload = step.payload or {}
        started_at = utcnow()

        if step.step_type == MissionStepType.WAIT.value:
            step.started_at = step.started_at or started_at
            step.updated_at = started_at
            mission.started_at = mission.started_at or started_at
            step.status = MissionStepStatus.WAITING.value
            mission.status = MissionStatus.WAITING.value
            mission.updated_at = started_at
            return

        if step.step_type == MissionStepType.MOVE_MARKER.value:
            task = self.task_manager.create_move_marker_task(
                payload["marker_name"],
                allow_interruption=bool(payload.get("allow_interruption", False)),
                mission_id=mission.id,
                mission_step_id=step.id,
            )
        elif step.step_type == MissionStepType.MOVE_COORDINATE.value:
            task = self.task_manager.create_move_location_task(
                payload["x"],
                payload["y"],
                payload["theta"],
                allow_interruption=bool(payload.get("allow_interruption", False)),
                mission_id=mission.id,
                mission_step_id=step.id,
            )
        elif step.step_type == MissionStepType.RETURN_TO_CHARGER.value:
            task = self.task_manager.return_to_charger(mission_id=mission.id, mission_step_id=step.id)
        elif step.step_type == MissionStepType.CANCEL_MOVE.value:
            task = self.task_manager.cancel_current_move(mission_id=mission.id, mission_step_id=step.id)
        elif step.step_type == MissionStepType.EMERGENCY_STOP.value:
            task = self.task_manager.emergency_stop(mission_id=mission.id, mission_step_id=step.id)
        else:
            raise ValueError(f"Unsupported mission step type '{step.step_type}'.")

        step.started_at = step.started_at or started_at
        step.updated_at = started_at
        mission.started_at = mission.started_at or started_at
        mission.status = MissionStatus.RUNNING.value
        mission.updated_at = started_at
        step.task_id = task["task_id"]
        step.compact_result = {"task": task}
        if task["status"] == TaskStatus.RUNNING.value:
            step.status = MissionStepStatus.RUNNING.value
            mission.status = MissionStatus.RUNNING.value
        elif task["status"] == TaskStatus.SUCCEEDED.value:
            self._mark_step_complete(mission, step, MissionStepStatus.SUCCEEDED, result=task)
        elif task["status"] == TaskStatus.CANCELED.value:
            self._mark_step_complete(mission, step, MissionStepStatus.CANCELED, result=task)
        else:
            self._mark_step_complete(mission, step, MissionStepStatus.FAILED, result=task, error_message=task.get("error_message"))
            mission.status = MissionStatus.FAILED.value
            mission.error_message = task.get("error_message")

    def _handle_wait_step(self, mission: MissionRecord, step: MissionStepRecord) -> None:
        wait_seconds = int((step.payload or {}).get("wait_seconds", 0))
        started_at = ensure_utc(step.started_at) if step.started_at else utcnow()
        step.started_at = started_at
        step.updated_at = utcnow()
        mission.status = MissionStatus.WAITING.value
        if utcnow() >= started_at + timedelta(seconds=wait_seconds):
            self._mark_step_complete(mission, step, MissionStepStatus.SUCCEEDED, result={"wait_seconds": wait_seconds})
            mission.status = MissionStatus.RUNNING.value

    def _maybe_replan(self, session, mission: MissionRecord, steps: list[MissionStepRecord], failed_step: MissionStepRecord) -> bool:
        if not mission.auto_replan or self.mission_planner is None or mission.replan_count >= self.settings.mission_max_replans:
            return False
        remaining_context = {
            "mission_id": mission.id,
            "user_request": mission.user_request,
            "failed_step": failed_step.description,
            "error_message": failed_step.error_message,
            "summary": mission.context_summary,
            "remaining_steps": [
                {
                    "step_index": step.step_index,
                    "step_type": step.step_type,
                    "description": step.description,
                    "payload": step.payload,
                    "status": step.status,
                }
                for step in steps
                if step.step_index >= failed_step.step_index and step.status != MissionStepStatus.SUCCEEDED.value
            ],
        }
        replanned = self.mission_planner.replan_steps(remaining_context)
        if replanned is None:
            return False

        for step in steps:
            if step.step_index >= failed_step.step_index and step.status not in {MissionStepStatus.SUCCEEDED.value, MissionStepStatus.CANCELED.value}:
                session.delete(step)
        session.flush()

        new_steps: list[MissionStepRecord] = []
        for offset, step_payload in enumerate(replanned["steps"], start=failed_step.step_index):
            new_steps.append(
                MissionStepRecord(
                    mission_id=mission.id,
                    step_index=offset,
                    step_type=step_payload["step_type"],
                    status=MissionStepStatus.PENDING.value,
                    description=step_payload.get("description") or self._step_description(step_payload),
                    payload=step_payload,
                )
            )
        session.add_all(new_steps)
        session.flush()
        mission.replan_count += 1
        mission.status = MissionStatus.RUNNING.value
        mission.error_message = replanned["response"]
        mission.current_step_index = failed_step.step_index
        return True

    def _load_steps(self, session, mission_id: str) -> list[MissionStepRecord]:
        return list(
            session.scalars(
                select(MissionStepRecord).where(MissionStepRecord.mission_id == mission_id).order_by(MissionStepRecord.step_index)
            ).all()
        )

    def _advance_mission(self, session, mission: MissionRecord, steps: list[MissionStepRecord]) -> bool:
        """Process the mission's current step once.

        Returns True when it made progress and the *next* current step should be
        processed immediately in this same poll pass (a step reached a terminal
        status, or a step transitioned into an active state that warrants an
        immediate re-check). Returns False when the mission is finished or is
        blocked waiting on the robot or a wall-clock timer — i.e. nothing more can
        happen until a later poll.

        Driving as far as possible per pass is what removes the poll-interval of
        dead time that used to sit between every step transition.
        """
        current_step = next((step for step in steps if step.status not in FINAL_STEP_STATUSES), None)
        if current_step is None:
            return False

        if current_step.status == MissionStepStatus.PENDING.value:
            # Release this session's read transaction, then atomically claim the
            # step. This both prevents a concurrent poller from dispatching the
            # same step and avoids self-contention with the task INSERT during the
            # (slow) move dispatch below.
            session.commit()
            if not self._claim_step(current_step.id):
                # Another poller already owns this step; leave it to them.
                return False
            current_step.status = MissionStepStatus.RUNNING.value
            try:
                self._dispatch_step(mission, current_step)
            except Exception as exc:
                current_step.status = MissionStepStatus.FAILED.value
                current_step.error_message = str(exc)
                current_step.completed_at = utcnow()
                mission.status = MissionStatus.FAILED.value
                mission.error_message = str(exc)
                return False
            # A wait is now WAITING (re-check its timer immediately), an instant
            # task is already terminal (advance), a move is RUNNING (re-check its
            # task immediately). Keep going.
            return True

        if current_step.status == MissionStepStatus.WAITING.value:
            self._handle_wait_step(mission, current_step)
            return current_step.status in FINAL_STEP_STATUSES

        if current_step.status == MissionStepStatus.RUNNING.value and not current_step.task_id:
            # Claimed for dispatch but never linked to a task: recover it so a
            # crashed poller can't strand the mission in RUNNING.
            self._recover_orphaned_step(current_step)
            return False

        if current_step.status == MissionStepStatus.RUNNING.value and current_step.task_id:
            task = self.task_manager.get_task(current_step.task_id)
            current_step.compact_result = {"task": task}
            current_step.updated_at = utcnow()
            mission.updated_at = utcnow()
            if task["status"] == TaskStatus.RUNNING.value:
                mission.status = MissionStatus.RUNNING.value
                return False
            if task["status"] == TaskStatus.SUCCEEDED.value:
                self._mark_step_complete(mission, current_step, MissionStepStatus.SUCCEEDED, result={"task": task})
                mission.status = MissionStatus.RUNNING.value
                return True
            if task["status"] == TaskStatus.CANCELED.value:
                self._mark_step_complete(mission, current_step, MissionStepStatus.CANCELED, result={"task": task}, error_message=task.get("error_message"))
                mission.status = MissionStatus.CANCELED.value
                mission.error_message = task.get("error_message") or "Mission step canceled."
                mission.completed_at = utcnow()
                return False
            current_step.error_message = task.get("error_message") or "Mission step failed."
            self._mark_step_complete(mission, current_step, MissionStepStatus.FAILED, result={"task": task}, error_message=current_step.error_message)
            mission.status = MissionStatus.FAILED.value
            mission.error_message = current_step.error_message
            if self._maybe_replan(session, mission, steps, current_step):
                # Replanning rewrote the remaining steps. Stop this pass and let
                # the next poll dispatch the new plan, so each replan gets a real
                # execution attempt instead of being re-evaluated (and possibly
                # re-failed) synchronously within one pass.
                mission.status = MissionStatus.RUNNING.value
                return False
            return False

        return False

    def poll_missions(self) -> None:
        session = self.session_factory()
        try:
            missions = list(
                session.scalars(
                    select(MissionRecord).where(MissionRecord.status.not_in(FINAL_MISSION_STATUSES)).order_by(MissionRecord.created_at.asc())
                ).all()
            )
            for mission in missions:
                steps = self._load_steps(session, mission.id)
                if not steps:
                    continue
                # Advance the mission as far as it can go this pass. The bound is a
                # safety backstop against an unexpected non-terminating transition;
                # real progress is monotonic (steps reach terminal status or get
                # dispatched at most once each, plus a bounded number of replans).
                max_iterations = len(steps) * (self.settings.mission_max_replans + 2) + 4
                for _ in range(max_iterations):
                    if not self._advance_mission(session, mission, steps):
                        break
                    # Steps may have been mutated (replan deletes/adds rows), so
                    # reload before selecting the next current step.
                    steps = self._load_steps(session, mission.id)

                # _advance may have mutated steps on the pass that returned False
                # (e.g. a replan rewrites the remaining steps), so reload before
                # settling the mission's final status against a fresh view.
                steps = self._load_steps(session, mission.id)
                if all(step.status in FINAL_STEP_STATUSES for step in steps) and mission.status not in FINAL_MISSION_STATUSES:
                    # Every step settled and no step forced FAILED/CANCELED -> the
                    # mission succeeded.
                    mission.status = MissionStatus.SUCCEEDED.value
                    mission.completed_at = mission.completed_at or utcnow()
                elif mission.status in {MissionStatus.FAILED.value, MissionStatus.CANCELED.value} and mission.completed_at is None:
                    mission.completed_at = utcnow()

                self._refresh_summary(mission, steps)
            session.commit()
        finally:
            session.close()

    async def start_polling(self) -> None:
        if self._polling_task is not None:
            return
        self._stop_event.clear()
        self._polling_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_missions()
            except Exception:
                pass
            await asyncio.sleep(self.settings.mission_poll_seconds)

    async def stop_polling(self) -> None:
        if self._polling_task is None:
            return
        self._stop_event.set()
        self._polling_task.cancel()
        try:
            await self._polling_task
        except asyncio.CancelledError:
            pass
        self._polling_task = None
