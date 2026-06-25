from app.config import Settings
from app.water.client import HttpWaterTransport, WaterRobotClient


def test_build_url_for_marker_move():
    settings = Settings(water_robot_host="192.168.10.10", water_http_port=9001)
    transport = HttpWaterTransport(settings)
    url = transport.build_url("/api/move", {"marker": "room_205", "uuid": "abc"})
    assert url == "http://192.168.10.10:9001/api/move?marker=room_205&uuid=abc"


def test_dry_run_move_returns_task_id():
    client = WaterRobotClient(Settings(water_dry_run=True))
    response = client.move_to_marker("room_205")
    assert response.status == "OK"
    assert response.task_id is not None
