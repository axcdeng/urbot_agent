from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.water.client import WaterRobotClient


def main() -> None:
    settings = Settings()
    client = WaterRobotClient(settings)
    payload = {
        "base_url": settings.http_base_url,
        "dry_run": settings.water_dry_run,
        "robot_status": client.get_robot_status().model_dump(mode="json"),
        "robot_info": client.get_robot_info().model_dump(mode="json"),
        "markers": client.query_marker_brief().model_dump(mode="json"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
