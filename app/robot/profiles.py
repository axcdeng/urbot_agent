"""Named robot profiles.

A profile bundles a robot address with the marker map that robot uses, so the
TUI's ``/ip <name>`` can switch both the endpoint and the active map in one go.

To give a profile its own distinct map, replace its ``markers`` with a dict in
the ``/api/markers/query_list`` shape (the same shape as ``DRY_MARKERS``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.water.client import DRY_MARKERS


@dataclass(frozen=True)
class RobotProfile:
    name: str
    host: str
    port: int
    markers: dict[str, Any]


# The secondary site currently shares the 1F waypoint map. Point this at its
# own dict (query_list shape) if/when the two maps diverge.
SECONDARY_MARKERS: dict[str, Any] = DRY_MARKERS


ROBOT_PROFILES: dict[str, RobotProfile] = {
    "primary": RobotProfile("primary", "10.1.17.225", 9001, DRY_MARKERS),
    "secondary": RobotProfile("secondary", "10.1.16.160", 9001, SECONDARY_MARKERS),
}
