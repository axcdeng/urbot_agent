from pathlib import Path

import pytest

from app.config import Settings
from app.db import create_session_factory, init_db
from app.robot.locations import LocationRegistry
from app.robot.safety import SafetyValidator
from app.robot.state_manager import StateManager
from app.robot.task_manager import TaskManager
from app.water.client import WaterRobotClient
from app.water.normalizer import normalize_marker_response
from app.water.schemas import WaterEnvelope


def build(tmp_path: Path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'charger.db'}", water_dry_run=True)
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    client = WaterRobotClient(settings)
    state_manager = StateManager(client)
    registry = LocationRegistry(session_factory, client)
    registry.sync_markers()
    safety = SafetyValidator(settings)
    manager = TaskManager(session_factory, client, state_manager, registry, safety)
    return registry, state_manager, manager


def test_normalizer_parses_charger_properties():
    envelope = WaterEnvelope(
        command="/api/markers/query_list",
        status="OK",
        results={
            "c": {
                "marker_name": "c",
                "key": 11,
                "floor": 1,
                "pose": {"position": {"x": 0, "y": 0, "z": 0}, "orientation": {"z": 0, "w": 1}},
                "properties": '{"cabin_key":"CAB1","chassis_key":"CH1","charging_pile_type":"up_charging_pile"}',
            }
        },
    )
    marker = normalize_marker_response(envelope)[0]
    assert marker["cabin_key"] == "CAB1"
    assert marker["chassis_key"] == "CH1"
    assert marker["charging_pile_type"] == "up_charging_pile"


def test_own_charger_resolves_to_chassis_match(tmp_path: Path):
    registry, state_manager, _ = build(tmp_path)
    identity = state_manager.get_device_identity()
    # Dry identity chassis_key matches charge_point_1F_40300423.
    assert registry.resolve_own_charger(identity) == "charge_point_1F_40300423"

    chargers = {c["marker_name"]: c for c in registry.list_chargers(identity)}
    assert chargers["charge_point_1F_40300423"]["charges_my_chassis"] is True
    # charge_point_1F_1 matches the attached cabin only.
    assert chargers["charge_point_1F_1"]["charges_my_cabin"] is True
    assert chargers["charge_point_1F_1"]["charges_my_chassis"] is False
    # A different robot's charger is not mine.
    assert chargers["charge_point_1F_40300165"]["is_mine"] is False


def test_return_to_charger_targets_own_charger(tmp_path: Path):
    _, _, manager = build(tmp_path)
    task = manager.return_to_charger()
    assert task["status"] == "running"
    assert task["requested_target"] == "charge_point_1F_40300423"


def test_move_to_foreign_charger_requires_confirmation(tmp_path: Path):
    _, _, manager = build(tmp_path)
    with pytest.raises(ValueError) as exc:
        manager.create_move_marker_task("charge_point_1F_40300165")
    assert "does not serve this robot" in str(exc.value)
    # With explicit confirmation the move proceeds.
    task = manager.create_move_marker_task("charge_point_1F_40300165", confirm_foreign_charger=True)
    assert task["status"] == "running"


def test_move_to_own_charger_is_allowed_without_confirmation(tmp_path: Path):
    _, _, manager = build(tmp_path)
    task = manager.create_move_marker_task("charge_point_1F_40300423")
    assert task["status"] == "running"
