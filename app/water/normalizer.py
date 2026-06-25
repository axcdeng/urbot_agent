from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import atan2
from typing import Any

from app.water.schemas import WaterEnvelope


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Pose:
    x: float
    y: float
    theta: float


@dataclass
class NormalizedRobotState:
    online: bool
    battery_percent: int | None
    charging: bool
    estop_state: bool
    current_floor: int | None
    current_pose: Pose | None
    current_target: str | None
    move_status: str
    running_status: str
    error_code: str | None
    last_updated: str
    raw_status: dict[str, Any] | None
    raw_power: dict[str, Any] | None
    raw_location: dict[str, Any] | None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.current_pose is not None:
            data["current_pose"] = asdict(self.current_pose)
        return data


def quaternion_to_theta(orientation: dict[str, Any] | None) -> float:
    if not orientation:
        return 0.0
    z = float(orientation.get("z", 0.0))
    w = float(orientation.get("w", 1.0))
    return 2 * atan2(z, w)


def normalize_marker_response(envelope: WaterEnvelope) -> list[dict[str, Any]]:
    if envelope.status != "OK" or not isinstance(envelope.results, dict):
        return []

    markers: list[dict[str, Any]] = []
    for marker_name, payload in envelope.results.items():
        pose = payload.get("pose") or {}
        position = pose.get("position") or {}
        orientation = pose.get("orientation") or {}
        markers.append(
            {
                "marker_name": marker_name,
                "floor": payload.get("floor"),
                "marker_type": payload.get("key"),
                "pose": {
                    "x": position.get("x"),
                    "y": position.get("y"),
                    "z": position.get("z"),
                    "theta": quaternion_to_theta(orientation),
                },
                "raw_payload": payload,
            }
        )
    markers.sort(key=lambda item: item["marker_name"])
    return markers


def normalize_map_response(envelope: WaterEnvelope) -> dict[str, Any]:
    results = envelope.results or {}
    if not isinstance(results, dict):
        results = {}
    return {
        "map_name": results.get("map_name") or results.get("hotel_id"),
        "floor": results.get("floor"),
        "info": results.get("info") or {},
        "raw": envelope.model_dump(mode="json"),
    }


def normalize_robot_state(
    status_envelope: WaterEnvelope,
    power_envelope: WaterEnvelope | None = None,
    location_envelope: WaterEnvelope | None = None,
) -> NormalizedRobotState:
    if status_envelope.status != "OK" or not isinstance(status_envelope.results, dict):
        return NormalizedRobotState(
            online=False,
            battery_percent=None,
            charging=False,
            estop_state=False,
            current_floor=None,
            current_pose=None,
            current_target=None,
            move_status="unknown",
            running_status="unknown",
            error_code=None,
            last_updated=utc_timestamp(),
            raw_status=status_envelope.model_dump(mode="json"),
            raw_power=power_envelope.model_dump(mode="json") if power_envelope else None,
            raw_location=location_envelope.model_dump(mode="json") if location_envelope else None,
            message=status_envelope.error_message or "Robot status unavailable.",
        )

    status = status_envelope.results or {}
    power = power_envelope.results if power_envelope and isinstance(power_envelope.results, dict) else {}
    location = location_envelope.results if location_envelope and isinstance(location_envelope.results, dict) else {}

    pose_source = location.get("current_pose") or status.get("current_pose")
    pose = None
    if isinstance(pose_source, dict):
        pose = Pose(
            x=float(pose_source.get("x", 0.0)),
            y=float(pose_source.get("y", 0.0)),
            theta=float(pose_source.get("theta", 0.0)),
        )

    battery_percent = status.get("power_percent")
    if battery_percent is None:
        battery_percent = power.get("battery_capacity")

    return NormalizedRobotState(
        online=True,
        battery_percent=int(battery_percent) if battery_percent is not None else None,
        charging=bool(status.get("charge_state") if status.get("charge_state") is not None else power.get("charger_connected_notice", False)),
        estop_state=bool(status.get("estop_state", False)),
        current_floor=location.get("current_floor") or status.get("current_floor"),
        current_pose=pose,
        current_target=status.get("move_target"),
        move_status=str(status.get("move_status", "unknown")),
        running_status=str(location.get("running_status") or status.get("running_status", "unknown")),
        error_code=status.get("error_code"),
        last_updated=utc_timestamp(),
        raw_status=status_envelope.model_dump(mode="json"),
        raw_power=power_envelope.model_dump(mode="json") if power_envelope else None,
        raw_location=location_envelope.model_dump(mode="json") if location_envelope else None,
    )


def compact_state_summary(state: NormalizedRobotState) -> dict[str, Any]:
    return {
        "online": state.online,
        "battery_percent": state.battery_percent,
        "charging": state.charging,
        "estop_state": state.estop_state,
        "current_floor": state.current_floor,
        "current_pose": asdict(state.current_pose) if state.current_pose else None,
        "current_target": state.current_target,
        "move_status": state.move_status,
        "running_status": state.running_status,
        "error_code": state.error_code,
        "last_updated": state.last_updated,
    }
