from app.config import Settings
from app.robot.state_manager import StateManager
from app.water.client import WaterRobotClient
from app.water.schemas import WaterClientError


def test_state_survives_location_endpoint_404():
    """A failing get_current_location must not make the robot look offline.

    Mirrors the deployed robot, where /api/get_current_location returns 404
    while robot_status and power work.
    """
    client = WaterRobotClient(Settings(water_dry_run=True))

    def boom():
        raise WaterClientError("404 Not Found")

    client.get_current_location = boom  # type: ignore[method-assign]

    state = StateManager(client).get_robot_state()
    assert state.online is True
    assert state.battery_percent is not None
    assert state.current_pose is not None  # falls back to status pose


def test_state_offline_when_status_fails():
    client = WaterRobotClient(Settings(water_dry_run=True))

    def boom():
        raise WaterClientError("connection refused")

    client.get_robot_status = boom  # type: ignore[method-assign]

    state = StateManager(client).get_robot_state()
    assert state.online is False
