from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models import LocationAlias, MarkerCache
from app.water.client import WaterRobotClient
from app.water.normalizer import normalize_marker_response


DEFAULT_ALIASES = {
    "charger": "charging",
    "charging station": "charging",
    "room 205": "room_205",
    "kitchen pickup": "kitchen_pickup",
    "front desk": "front_desk",
}


@dataclass
class ResolvedLocation:
    requested_name: str
    marker_name: str | None
    source: str


class LocationRegistry:
    def __init__(self, session_factory: sessionmaker, client: WaterRobotClient):
        self.session_factory = session_factory
        self.client = client
        self._ensure_default_aliases()

    def _ensure_default_aliases(self) -> None:
        session = self.session_factory()
        try:
            for alias, marker_name in DEFAULT_ALIASES.items():
                existing = session.get(LocationAlias, alias)
                if existing is None:
                    session.add(LocationAlias(alias=alias, marker_name=marker_name))
            session.commit()
        finally:
            session.close()

    def sync_markers(self) -> list[dict[str, Any]]:
        envelope = self.client.query_markers()
        markers = normalize_marker_response(envelope)
        session = self.session_factory()
        try:
            session.query(MarkerCache).delete()
            for marker in markers:
                session.add(
                    MarkerCache(
                        marker_name=marker["marker_name"],
                        floor=marker["floor"],
                        marker_type=marker["marker_type"],
                        pose=marker["pose"],
                        raw_payload=marker["raw_payload"],
                    )
                )
            session.commit()
        finally:
            session.close()
        return markers

    def list_locations(self) -> dict[str, Any]:
        session = self.session_factory()
        try:
            markers = session.scalars(select(MarkerCache).order_by(MarkerCache.marker_name)).all()
            aliases = session.scalars(select(LocationAlias).order_by(LocationAlias.alias)).all()
            return {
                "markers": [
                    {
                        "marker_name": marker.marker_name,
                        "floor": marker.floor,
                        "marker_type": marker.marker_type,
                        "pose": marker.pose,
                    }
                    for marker in markers
                ],
                "aliases": [{"alias": alias.alias, "marker_name": alias.marker_name} for alias in aliases],
            }
        finally:
            session.close()

    def resolve_location(self, name: str) -> ResolvedLocation:
        normalized = name.strip().lower()
        session = self.session_factory()
        try:
            alias = session.get(LocationAlias, normalized)
            if alias is not None:
                return ResolvedLocation(requested_name=name, marker_name=alias.marker_name, source="alias")
            marker = session.get(MarkerCache, normalized)
            if marker is not None:
                return ResolvedLocation(requested_name=name, marker_name=marker.marker_name, source="marker")
        finally:
            session.close()
        return ResolvedLocation(requested_name=name, marker_name=None, source="unknown")

    def add_alias(self, alias: str, marker_name: str) -> dict[str, str]:
        session = self.session_factory()
        try:
            marker = session.get(MarkerCache, marker_name)
            if marker is None:
                raise ValueError(f"Marker '{marker_name}' does not exist in cache.")
            normalized_alias = alias.strip().lower()
            session.merge(LocationAlias(alias=normalized_alias, marker_name=marker_name))
            session.commit()
            return {"alias": normalized_alias, "marker_name": marker_name}
        finally:
            session.close()

    def delete_alias(self, alias: str) -> None:
        session = self.session_factory()
        try:
            record = session.get(LocationAlias, alias.strip().lower())
            if record is None:
                raise ValueError(f"Alias '{alias}' does not exist.")
            session.delete(record)
            session.commit()
        finally:
            session.close()
