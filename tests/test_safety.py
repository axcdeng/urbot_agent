from app.config import Settings
from app.robot.safety import SafetyValidator
from app.water.normalizer import NormalizedRobotState


def make_state(**overrides):
    base = {
        "online": True,
        "battery_percent": 80,
        "charging": False,
        "estop_state": False,
        "current_floor": 1,
        "current_pose": None,
        "current_target": None,
        "move_status": "idle",
        "running_status": "idle",
        "error_code": "00000000",
        "last_updated": "2026-06-18T00:00:00Z",
        "raw_status": None,
        "raw_power": None,
        "raw_location": None,
        "message": "",
    }
    base.update(overrides)
    return NormalizedRobotState(**base)


def test_low_battery_blocks_move():
    validator = SafetyValidator(Settings(min_move_battery_percent=20))
    decision = validator.validate_move(make_state(battery_percent=10))
    assert decision.allowed is False
    assert decision.code == "low_battery"


def test_unknown_target_blocks_move():
    validator = SafetyValidator(Settings())
    decision = validator.validate_move(make_state(), target_exists=False)
    assert decision.allowed is False
    assert decision.code == "unknown_target"
