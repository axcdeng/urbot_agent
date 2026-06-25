from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.db import create_session_factory, init_db
from app.robot.locations import LocationRegistry
from app.water.client import WaterRobotClient


def main() -> None:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    registry = LocationRegistry(session_factory, WaterRobotClient(settings))
    markers = registry.sync_markers()
    print(json.dumps({"count": len(markers), "markers": markers}, indent=2))


if __name__ == "__main__":
    main()
