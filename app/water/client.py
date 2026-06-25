from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import httpx

from app.config import Settings
from app.water.schemas import WaterClientError, WaterEnvelope


DRY_MARKERS = {
    "charging": {
        "floor": 1,
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
        "marker_name": "charging",
        "key": 11,
    },
    "room_205": {
        "floor": 2,
        "pose": {
            "position": {"x": 12.0, "y": 4.0, "z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
        "marker_name": "room_205",
        "key": 0,
    },
    "kitchen_pickup": {
        "floor": 1,
        "pose": {
            "position": {"x": 3.5, "y": 8.25, "z": 0.0},
            "orientation": {"w": 0.9239, "x": 0.0, "y": 0.0, "z": 0.3827},
        },
        "marker_name": "kitchen_pickup",
        "key": 0,
    },
    "front_desk": {
        "floor": 1,
        "pose": {
            "position": {"x": -4.0, "y": 2.0, "z": 0.0},
            "orientation": {"w": 0.7071, "x": 0.0, "y": 0.0, "z": 0.7071},
        },
        "marker_name": "front_desk",
        "key": 0,
    },
}


@dataclass
class HttpWaterTransport:
    settings: Settings

    def build_url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.settings.http_base_url}{path}"
        if params:
            return f"{url}?{urlencode(params)}"
        return url

    def send(self, path: str, params: dict[str, Any] | None = None) -> WaterEnvelope:
        url = self.build_url(path, params)
        try:
            response = httpx.get(url, timeout=self.settings.water_timeout_seconds)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise WaterClientError(f"Timed out calling {path}") from exc
        except httpx.HTTPError as exc:
            raise WaterClientError(f"HTTP error calling {path}: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise WaterClientError(f"Non-JSON response calling {path}") from exc
        return WaterEnvelope.model_validate(payload)


@dataclass
class TcpWaterTransportPlaceholder:
    settings: Settings

    def subscribe(self, topic: str, frequency: float) -> dict[str, Any]:
        return {
            "implemented": False,
            "topic": topic,
            "frequency": frequency,
            "message": "TCP callback support is intentionally stubbed in v1.",
        }


class WaterRobotClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.http_transport = HttpWaterTransport(settings)
        self.tcp_transport = TcpWaterTransportPlaceholder(settings)
        self._dry_state = {
            "soft_estop": False,
            "hard_estop": False,
            "battery_percent": 82,
            "charging": False,
            "current_floor": 1,
            "current_pose": {"x": 1.0, "y": 1.0, "theta": 0.0},
            "current_marker": "front_desk",
            "move_target": "",
            "move_status": "idle",
            "running_status": "idle",
            "error_code": "00000000",
            "pending_ticks": 0,
            "robot_task_id": None,
            "target_pose": None,
            "move_history": [],
        }

    @property
    def dry_run(self) -> bool:
        return self.settings.water_dry_run

    def get_transport_url(self, path: str, params: dict[str, Any] | None = None) -> str:
        return self.http_transport.build_url(path, params)

    def _request(self, path: str, params: dict[str, Any] | None = None) -> WaterEnvelope:
        if self.dry_run:
            return self._dry_response(path, params or {})
        return self.http_transport.send(path, params)

    def _dry_status_results(self) -> dict[str, Any]:
        if self._dry_state["pending_ticks"] > 0:
            self._dry_state["pending_ticks"] -= 1
            self._dry_state["move_status"] = "running"
            self._dry_state["running_status"] = "running"
        elif self._dry_state["move_status"] == "running":
            self._dry_state["move_status"] = "succeeded"
            self._dry_state["running_status"] = "idle"
            self._dry_state["current_marker"] = self._dry_state["move_target"] or self._dry_state["current_marker"]
            target_pose = self._dry_state["target_pose"]
            if target_pose:
                self._dry_state["current_pose"] = target_pose
            self._dry_state["move_target"] = ""
            self._dry_state["robot_task_id"] = None
        return {
            "move_target": self._dry_state["move_target"],
            "move_status": self._dry_state["move_status"],
            "running_status": self._dry_state["running_status"],
            "move_retry_times": 0,
            "charge_state": self._dry_state["charging"],
            "soft_estop_state": self._dry_state["soft_estop"],
            "hard_estop_state": self._dry_state["hard_estop"],
            "estop_state": self._dry_state["soft_estop"] or self._dry_state["hard_estop"],
            "power_percent": self._dry_state["battery_percent"],
            "current_pose": self._dry_state["current_pose"],
            "current_floor": self._dry_state["current_floor"],
            "chargepile_id": "0",
            "error_code": self._dry_state["error_code"],
        }

    def _move_to_marker_pose(self, marker_name: str) -> dict[str, Any] | None:
        marker = DRY_MARKERS.get(marker_name)
        if not marker:
            return None
        position = marker["pose"]["position"]
        return {"x": position["x"], "y": position["y"], "theta": 0.0}

    def _dry_response(self, path: str, params: dict[str, Any]) -> WaterEnvelope:
        if path == "/api/robot_status":
            return WaterEnvelope(command=path, status="OK", results=self._dry_status_results())

        if path == "/api/robot_info":
            return WaterEnvelope(command=path, status="OK", results={"product_id": "WATER-DRYRUN-001"})

        if path == "/api/get_power_status":
            return WaterEnvelope(
                command=path,
                status="OK",
                results={
                    "battery_capacity": self._dry_state["battery_percent"],
                    "battery_current": -0.1,
                    "battery_voltage": 29.5,
                    "charge_voltage": 28.9,
                    "charger_connected_notice": self._dry_state["charging"],
                    "head_current": 0,
                },
            )

        if path == "/api/get_battery_status":
            return WaterEnvelope(
                command=path,
                status="OK",
                results={
                    "battery_initial_capacity": 18,
                    "loop_count": 27,
                    "soh": 98,
                    "temperature": 29,
                },
            )

        if path == "/api/get_current_location":
            marker_name = self._dry_state["current_marker"]
            near_markers = []
            if marker_name:
                near_markers.append({"distance": 0.1, "key": DRY_MARKERS[marker_name]["key"], "marker_name": marker_name})
            return WaterEnvelope(
                command=path,
                status="OK",
                results={
                    "current_floor": self._dry_state["current_floor"],
                    "current_pose": self._dry_state["current_pose"],
                    "near_markers": near_markers,
                    "running_status": self._dry_state["running_status"],
                },
            )

        if path == "/api/markers/query_list":
            return WaterEnvelope(command=path, status="OK", results=copy.deepcopy(DRY_MARKERS))

        if path == "/api/markers/query_brief":
            brief = {name: f"{payload['key']}-{payload['floor']}" for name, payload in DRY_MARKERS.items()}
            return WaterEnvelope(command=path, status="OK", results=brief)

        if path == "/api/map/get_current_map":
            return WaterEnvelope(
                command=path,
                status="OK",
                results={
                    "map_name": "demo_map",
                    "floor": str(self._dry_state["current_floor"]),
                    "info": {"resolution": 0.05, "width": 1024, "height": 768, "origin_x": -10.0, "origin_y": -10.0},
                },
            )

        if path == "/api/move":
            task_id = uuid4().hex.upper()
            target_marker = params.get("marker")
            target_location = params.get("location")
            if self._dry_state["soft_estop"] or self._dry_state["hard_estop"]:
                return WaterEnvelope(command=path, status="REQUEST_DENIED", error_message="Robot is estopped")
            self._dry_state["robot_task_id"] = task_id
            self._dry_state["move_target"] = target_marker or ""
            self._dry_state["move_status"] = "running"
            self._dry_state["running_status"] = "running"
            self._dry_state["pending_ticks"] = 2
            self._dry_state["target_pose"] = self._move_to_marker_pose(target_marker) if target_marker else None
            if target_location:
                x_str, y_str, theta_str = target_location.split(",")
                self._dry_state["target_pose"] = {"x": float(x_str), "y": float(y_str), "theta": float(theta_str)}
            self._dry_state["move_history"].append({"task_id": task_id, "target": target_marker or target_location})
            return WaterEnvelope(command=path, status="OK", task_id=task_id)

        if path == "/api/move/cancel":
            self._dry_state["move_status"] = "canceled"
            self._dry_state["running_status"] = "idle"
            self._dry_state["move_target"] = ""
            self._dry_state["pending_ticks"] = 0
            self._dry_state["robot_task_id"] = None
            return WaterEnvelope(command=path, status="OK")

        if path == "/api/estop":
            enabled = str(params.get("flag", "")).lower() == "true"
            self._dry_state["soft_estop"] = enabled
            if enabled:
                self._dry_state["move_status"] = "canceled"
                self._dry_state["running_status"] = "idle"
                self._dry_state["pending_ticks"] = 0
            return WaterEnvelope(command=path, status="OK")

        if path == "/api/request_data":
            return WaterEnvelope(command=path, status="OK")

        return WaterEnvelope(command=path, status="INVALID_REQUEST", error_message=f"Unsupported dry-run path: {path}")

    def get_robot_status(self) -> WaterEnvelope:
        return self._request("/api/robot_status")

    def get_robot_info(self) -> WaterEnvelope:
        return self._request("/api/robot_info")

    def get_power_status(self) -> WaterEnvelope:
        return self._request("/api/get_power_status")

    def get_battery_status(self) -> WaterEnvelope:
        return self._request("/api/get_battery_status")

    def get_current_location(self) -> WaterEnvelope:
        return self._request("/api/get_current_location")

    def query_markers(self) -> WaterEnvelope:
        return self._request("/api/markers/query_list")

    def query_marker_brief(self) -> WaterEnvelope:
        return self._request("/api/markers/query_brief")

    def get_current_map(self) -> WaterEnvelope:
        return self._request("/api/map/get_current_map")

    def move_to_marker(self, marker_name: str, uuid: str | None = None) -> WaterEnvelope:
        params = {"marker": marker_name}
        if uuid:
            params["uuid"] = uuid
        return self._request("/api/move", params)

    def move_to_location(self, x: float, y: float, theta: float, uuid: str | None = None) -> WaterEnvelope:
        params = {"location": f"{x},{y},{theta}"}
        if uuid:
            params["uuid"] = uuid
        return self._request("/api/move", params)

    def cancel_move(self) -> WaterEnvelope:
        return self._request("/api/move/cancel")

    def set_estop(self, enabled: bool) -> WaterEnvelope:
        return self._request("/api/estop", {"flag": str(enabled).lower()})

    def request_realtime_data(self, topic: str, frequency: float = 1.0) -> dict[str, Any]:
        return self.tcp_transport.subscribe(topic, frequency)
