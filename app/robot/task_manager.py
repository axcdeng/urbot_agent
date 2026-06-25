from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models import EventLog, TaskRecord, TaskStatus, TaskType
from app.robot.locations import LocationRegistry
from app.robot.safety import SafetyValidator
from app.robot.state_manager import StateManager
from app.water.client import WaterRobotClient


logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TaskResult:
    record: TaskRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.record.id,
            "task_type": self.record.task_type,
            "status": self.record.status,
            "requested_target": self.record.requested_target,
            "robot_task_id": self.record.robot_task_id,
            "mission_id": self.record.mission_id,
            "mission_step_id": self.record.mission_step_id,
            "error_message": self.record.error_message,
            "created_at": self.record.created_at.isoformat() if self.record.created_at else None,
            "updated_at": self.record.updated_at.isoformat() if self.record.updated_at else None,
            "completed_at": self.record.completed_at.isoformat() if self.record.completed_at else None,
        }


class TaskManager:
    def __init__(
        self,
        session_factory: sessionmaker,
        client: WaterRobotClient,
        state_manager: StateManager,
        location_registry: LocationRegistry,
        safety: SafetyValidator,
    ):
        self.session_factory = session_factory
        self.client = client
        self.state_manager = state_manager
        self.location_registry = location_registry
        self.safety = safety
        self._polling_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    def _log(self, level: str, event_type: str, message: str, payload: dict[str, Any] | None = None) -> None:
        logger.log(getattr(logging, level.upper(), logging.INFO), "%s: %s", event_type, message)
        session = self.session_factory()
        try:
            session.add(EventLog(level=level.upper(), event_type=event_type, message=message, payload=payload))
            session.commit()
        except Exception:
            session.rollback()
            logger.debug("Skipping event log persistence for %s due to a transient database error.", event_type, exc_info=True)
        finally:
            session.close()

    def _new_task(
        self,
        session,
        task_type: TaskType,
        requested_target: str | None,
        payload: dict[str, Any] | None,
        allow_interruption: bool = False,
        mission_id: str | None = None,
        mission_step_id: str | None = None,
    ):
        record = TaskRecord(
            task_type=task_type.value,
            status=TaskStatus.CREATED.value,
            requested_target=requested_target,
            command_payload=payload,
            allow_interruption=allow_interruption,
            mission_id=mission_id,
            mission_step_id=mission_step_id,
        )
        session.add(record)
        session.flush()
        self._log("info", "task_created", f"Created task {record.id}", {"task_type": task_type.value, "target": requested_target})
        return record

    def _set_status(self, record: TaskRecord, status: TaskStatus, *, error_message: str | None = None, raw_response: dict[str, Any] | None = None, robot_task_id: str | None = None) -> None:
        record.status = status.value
        record.error_message = error_message
        record.raw_robot_response = raw_response
        record.updated_at = utcnow()
        if robot_task_id:
            record.robot_task_id = robot_task_id
        if status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED}:
            record.completed_at = utcnow()

    def _raise_validation(self, session, record: TaskRecord, reason: str) -> None:
        record.status = TaskStatus.FAILED.value
        record.error_message = reason
        record.updated_at = utcnow()
        record.completed_at = utcnow()
        session.commit()
        raise ValueError(reason)

    def get_task(self, task_id: str) -> dict[str, Any]:
        session = self.session_factory()
        try:
            record = session.get(TaskRecord, task_id)
            if record is None:
                raise ValueError(f"Task '{task_id}' not found.")
            return TaskResult(record).to_dict()
        finally:
            session.close()

    def list_tasks(self) -> list[dict[str, Any]]:
        session = self.session_factory()
        try:
            records = session.scalars(select(TaskRecord).order_by(TaskRecord.created_at.desc())).all()
            return [TaskResult(record).to_dict() for record in records]
        finally:
            session.close()

    def create_move_marker_task(
        self,
        marker_name: str,
        allow_interruption: bool = False,
        mission_id: str | None = None,
        mission_step_id: str | None = None,
    ) -> dict[str, Any]:
        session = self.session_factory()
        try:
            resolved = self.location_registry.resolve_location(marker_name)
            record = self._new_task(
                session,
                TaskType.MOVE_MARKER,
                marker_name,
                {"marker_name": marker_name},
                allow_interruption,
                mission_id=mission_id,
                mission_step_id=mission_step_id,
            )
            state = self.state_manager.get_robot_state()
            decision = self.safety.validate_move(
                state,
                target_exists=resolved.marker_name is not None,
                allow_interruption=allow_interruption,
            )
            if not decision.allowed:
                self._raise_validation(session, record, decision.reason)
            record.status = TaskStatus.VALIDATED.value
            response = self.client.move_to_marker(resolved.marker_name or marker_name, uuid=record.id)
            if response.status != "OK":
                self._set_status(record, TaskStatus.FAILED, error_message=response.error_message or response.status, raw_response=response.model_dump(mode="json"))
                session.commit()
                raise ValueError(record.error_message)
            self._set_status(record, TaskStatus.RUNNING, raw_response=response.model_dump(mode="json"), robot_task_id=response.task_id)
            session.commit()
            self._log("info", "move_marker", f"Sent marker task {record.id}", {"marker_name": resolved.marker_name})
            return TaskResult(record).to_dict()
        finally:
            session.close()

    def create_move_location_task(
        self,
        x: float,
        y: float,
        theta: float,
        allow_interruption: bool = False,
        mission_id: str | None = None,
        mission_step_id: str | None = None,
    ) -> dict[str, Any]:
        session = self.session_factory()
        try:
            record = self._new_task(
                session,
                TaskType.MOVE_COORDINATE,
                f"{x},{y},{theta}",
                {"x": x, "y": y, "theta": theta},
                allow_interruption,
                mission_id=mission_id,
                mission_step_id=mission_step_id,
            )
            state = self.state_manager.get_robot_state()
            decision = self.safety.validate_move(state, target_exists=True, allow_interruption=allow_interruption)
            if not decision.allowed:
                self._raise_validation(session, record, decision.reason)
            record.status = TaskStatus.VALIDATED.value
            response = self.client.move_to_location(x, y, theta, uuid=record.id)
            if response.status != "OK":
                self._set_status(record, TaskStatus.FAILED, error_message=response.error_message or response.status, raw_response=response.model_dump(mode="json"))
                session.commit()
                raise ValueError(record.error_message)
            self._set_status(record, TaskStatus.RUNNING, raw_response=response.model_dump(mode="json"), robot_task_id=response.task_id)
            session.commit()
            self._log("info", "move_location", f"Sent coordinate task {record.id}", {"x": x, "y": y, "theta": theta})
            return TaskResult(record).to_dict()
        finally:
            session.close()

    def cancel_current_move(self, mission_id: str | None = None, mission_step_id: str | None = None) -> dict[str, Any]:
        session = self.session_factory()
        try:
            record = self._new_task(session, TaskType.CANCEL_MOVE, None, None, mission_id=mission_id, mission_step_id=mission_step_id)
            response = self.client.cancel_move()
            if response.status != "OK":
                self._set_status(record, TaskStatus.FAILED, error_message=response.error_message or response.status, raw_response=response.model_dump(mode="json"))
                session.commit()
                raise ValueError(record.error_message)
            self._set_status(record, TaskStatus.SUCCEEDED, raw_response=response.model_dump(mode="json"))
            session.commit()
            self._log("warning", "cancel_move", f"Canceled move via task {record.id}")
            return TaskResult(record).to_dict()
        finally:
            session.close()

    def return_to_charger(self, mission_id: str | None = None, mission_step_id: str | None = None) -> dict[str, Any]:
        return self.create_move_marker_task("charger", mission_id=mission_id, mission_step_id=mission_step_id)

    def emergency_stop(self, mission_id: str | None = None, mission_step_id: str | None = None) -> dict[str, Any]:
        session = self.session_factory()
        try:
            record = self._new_task(
                session,
                TaskType.EMERGENCY_STOP,
                None,
                {"enabled": True},
                mission_id=mission_id,
                mission_step_id=mission_step_id,
            )
            response = self.client.set_estop(True)
            if response.status != "OK":
                self._set_status(record, TaskStatus.FAILED, error_message=response.error_message or response.status, raw_response=response.model_dump(mode="json"))
                session.commit()
                raise ValueError(record.error_message)
            self._set_status(record, TaskStatus.SUCCEEDED, raw_response=response.model_dump(mode="json"))
            session.commit()
            self._log("warning", "estop", f"Emergency stop triggered via task {record.id}")
            return TaskResult(record).to_dict()
        finally:
            session.close()

    def release_emergency_stop(self, confirmed: bool, mission_id: str | None = None, mission_step_id: str | None = None) -> dict[str, Any]:
        decision = self.safety.validate_release_estop(confirmed)
        if not decision.allowed:
            raise ValueError(decision.reason)
        session = self.session_factory()
        try:
            record = self._new_task(
                session,
                TaskType.RELEASE_EMERGENCY_STOP,
                None,
                {"enabled": False, "confirmed": confirmed},
                mission_id=mission_id,
                mission_step_id=mission_step_id,
            )
            response = self.client.set_estop(False)
            if response.status != "OK":
                self._set_status(record, TaskStatus.FAILED, error_message=response.error_message or response.status, raw_response=response.model_dump(mode="json"))
                session.commit()
                raise ValueError(record.error_message)
            self._set_status(record, TaskStatus.SUCCEEDED, raw_response=response.model_dump(mode="json"))
            session.commit()
            self._log("info", "release_estop", f"Released emergency stop via task {record.id}")
            return TaskResult(record).to_dict()
        finally:
            session.close()

    def poll_active_tasks(self) -> None:
        state = self.state_manager.get_robot_state()
        session = self.session_factory()
        try:
            active_records = session.scalars(
                select(TaskRecord).where(TaskRecord.status.in_([TaskStatus.RUNNING.value, TaskStatus.SENT_TO_ROBOT.value]))
            ).all()
            for record in active_records:
                if record.task_type not in {TaskType.MOVE_MARKER.value, TaskType.MOVE_COORDINATE.value, TaskType.RETURN_TO_CHARGER.value}:
                    continue
                if state.move_status == "running":
                    record.status = TaskStatus.RUNNING.value
                    record.updated_at = utcnow()
                    continue
                if state.move_status == "succeeded":
                    self._set_status(record, TaskStatus.SUCCEEDED)
                elif state.move_status == "failed":
                    self._set_status(record, TaskStatus.FAILED, error_message="Robot reported move failure.")
                elif state.move_status == "canceled":
                    self._set_status(record, TaskStatus.CANCELED)
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
                self.poll_active_tasks()
            except Exception as exc:  # pragma: no cover
                self._log("error", "poll_error", "Task poll loop failed.", {"error": str(exc)})
            await asyncio.sleep(2)

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
