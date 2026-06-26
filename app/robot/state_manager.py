from __future__ import annotations

from typing import Any

from app.water.client import WaterRobotClient
from app.water.normalizer import compact_state_summary, normalize_map_response, normalize_robot_state
from app.water.schemas import WaterClientError


class StateManager:
    def __init__(self, client: WaterRobotClient):
        self.client = client

    def _optional_call(self, fn):
        """Best-effort supplementary call; returns None if the endpoint fails.

        Some firmware/units don't implement every read endpoint (e.g.
        /api/get_current_location can 404), and those are supplementary —
        robot_status already carries pose/floor. A failure here must not
        make the robot look offline or block moves.
        """
        try:
            return fn()
        except WaterClientError:
            return None

    def get_robot_state(self):
        # robot_status is authoritative for online/e-stop/move state; if it
        # fails the robot is genuinely unreachable.
        try:
            status = self.client.get_robot_status()
        except WaterClientError as exc:
            return normalize_robot_state(
                type("OfflineEnvelope", (), {"status": "UNKNOWN_ERROR", "results": None, "error_message": str(exc), "model_dump": lambda self, mode="json": {"status": "UNKNOWN_ERROR", "error_message": str(exc)}})()
            )
        power = self._optional_call(self.client.get_power_status)
        location = self._optional_call(self.client.get_current_location)
        return normalize_robot_state(status, power, location)

    def get_compact_robot_state(self) -> dict[str, Any]:
        return compact_state_summary(self.get_robot_state())

    def get_robot_info(self) -> dict[str, Any]:
        return self.client.get_robot_info().model_dump(mode="json")

    def get_battery_status(self) -> dict[str, Any]:
        return self.client.get_battery_status().model_dump(mode="json")

    def get_robot_location(self) -> dict[str, Any]:
        return self.client.get_current_location().model_dump(mode="json")

    def get_robot_map(self) -> dict[str, Any]:
        return normalize_map_response(self.client.get_current_map())

    def get_robot_markers(self) -> dict[str, Any]:
        return self.client.query_markers().model_dump(mode="json")
