from app.water.normalizer import normalize_marker_response, normalize_robot_state
from app.water.schemas import WaterEnvelope


def test_normalize_robot_state_from_documented_payload():
    status = WaterEnvelope(
        command="/api/robot_status",
        status="OK",
        results={
            "move_target": "room_205",
            "move_status": "running",
            "running_status": "running",
            "charge_state": False,
            "estop_state": False,
            "power_percent": 66,
            "current_pose": {"x": 1.0, "y": 2.0, "theta": 0.5},
            "current_floor": 2,
            "error_code": "00000000",
        },
    )
    power = WaterEnvelope(command="/api/get_power_status", status="OK", results={"battery_capacity": 66, "charger_connected_notice": False})
    location = WaterEnvelope(command="/api/get_current_location", status="OK", results={"current_floor": 2, "current_pose": {"x": 1.0, "y": 2.0, "theta": 0.5}, "running_status": "running"})
    normalized = normalize_robot_state(status, power, location)
    assert normalized.online is True
    assert normalized.battery_percent == 66
    assert normalized.current_floor == 2
    assert normalized.current_target == "room_205"


def test_normalize_marker_query_response():
    response = WaterEnvelope(
        command="/api/markers/query_list",
        status="OK",
        results={
            "front_desk": {
                "floor": 1,
                "pose": {
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
                "marker_name": "front_desk",
                "key": 0,
            }
        },
    )
    markers = normalize_marker_response(response)
    assert markers[0]["marker_name"] == "front_desk"
    assert markers[0]["floor"] == 1
