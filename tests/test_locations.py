from pathlib import Path

from app.config import Settings
from app.db import create_session_factory, init_db
from app.robot.locations import LocationRegistry
from app.water.client import WaterRobotClient


def test_alias_resolution_and_marker_sync(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    settings = Settings(database_url=db_url, water_dry_run=True)
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    registry = LocationRegistry(session_factory, WaterRobotClient(settings))
    registry.sync_markers()
    resolved = registry.resolve_location("charger")
    assert resolved.marker_name == "charging"
    registry.add_alias("lobby drop", "front_desk")
    resolved_alias = registry.resolve_location("lobby drop")
    assert resolved_alias.marker_name == "front_desk"


def test_unknown_location_returns_none(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    settings = Settings(database_url=db_url, water_dry_run=True)
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    registry = LocationRegistry(session_factory, WaterRobotClient(settings))
    registry.sync_markers()
    assert registry.resolve_location("mars").marker_name is None
