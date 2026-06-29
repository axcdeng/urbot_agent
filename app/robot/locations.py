from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.models import LocationAlias, MarkerCache
from app.water.client import WaterRobotClient
from app.water.normalizer import normalize_marker_response, parse_marker_properties
from app.water.schemas import WaterEnvelope


# WATER marker `key` (point type) for a charging dock.
CHARGER_MARKER_TYPE = 11


# Friendly names mapped to canonical marker names on the deployed 1F map.
# Alias targets must be real marker names; resolution verifies they exist.
DEFAULT_ALIASES = {
    "charger": "charge_point_1F_1",
    "charging station": "charge_point_1F_1",
    "front desk": "front_desk",
    "reception": "toReception",
    "meeting room": "Meetingroom",
    "kitchen": "Kitchen",
    "security": "securitycheck",
    "security check": "securitycheck",
}


@dataclass
class ResolvedLocation:
    requested_name: str
    marker_name: str | None
    source: str


def _charger_keys(raw_payload: dict[str, Any] | None) -> dict[str, Any]:
    props = parse_marker_properties((raw_payload or {}).get("properties"))
    return {
        "cabin_key": props.get("cabin_key"),
        "chassis_key": props.get("chassis_key"),
        "charging_pile_type": props.get("charging_pile_type"),
    }


def _charger_match(keys: dict[str, Any], identity: dict[str, Any] | None) -> dict[str, bool]:
    """Does a charger (its parsed keys) serve THIS robot (its identity)?"""
    identity = identity or {}
    my_chassis = identity.get("chassis_key")
    my_cabin = identity.get("cabin_key")
    charges_my_chassis = bool(my_chassis and keys.get("chassis_key") == my_chassis)
    charges_my_cabin = bool(my_cabin and keys.get("cabin_key") == my_cabin)
    return {
        "charges_my_chassis": charges_my_chassis,
        "charges_my_cabin": charges_my_cabin,
        "is_mine": charges_my_chassis or charges_my_cabin,
    }


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

    def _store_markers(self, markers: list[dict[str, Any]]) -> None:
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

    def sync_markers(self) -> list[dict[str, Any]]:
        envelope = self.client.query_markers()
        markers = normalize_marker_response(envelope)
        self._store_markers(markers)
        return markers

    def load_marker_map(self, markers: dict[str, Any]) -> list[dict[str, Any]]:
        """Replace the marker cache from an in-memory map (query_list shape).

        Used when switching robot profiles so the agent resolves against the
        chosen map regardless of what (if anything) the robot reports.
        """
        envelope = WaterEnvelope(command="/api/markers/query_list", status="OK", results=copy.deepcopy(markers))
        markers_norm = normalize_marker_response(envelope)
        self._store_markers(markers_norm)
        self._ensure_default_aliases()
        return markers_norm

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
                        **(
                            _charger_keys(marker.raw_payload)
                            if marker.marker_type == CHARGER_MARKER_TYPE
                            else {}
                        ),
                    }
                    for marker in markers
                ],
                "aliases": [{"alias": alias.alias, "marker_name": alias.marker_name} for alias in aliases],
            }
        finally:
            session.close()

    def list_chargers(self, identity: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """All charging-dock markers with their keys and how they relate to this robot."""
        session = self.session_factory()
        try:
            rows = session.scalars(
                select(MarkerCache)
                .where(MarkerCache.marker_type == CHARGER_MARKER_TYPE)
                .order_by(MarkerCache.marker_name)
            ).all()
            chargers = []
            for marker in rows:
                keys = _charger_keys(marker.raw_payload)
                chargers.append(
                    {
                        "marker_name": marker.marker_name,
                        "floor": marker.floor,
                        **keys,
                        **_charger_match(keys, identity),
                    }
                )
            return chargers
        finally:
            session.close()

    def resolve_own_charger(self, identity: dict[str, Any] | None) -> str | None:
        """The marker name of this robot's own charger.

        Prefers a charger that charges this robot's chassis (the base needs to
        dock to recharge); falls back to one that charges its attached cabin.
        Returns None if identity is unknown or no charger matches.
        """
        chargers = self.list_chargers(identity)
        for charger in chargers:
            if charger["charges_my_chassis"]:
                return charger["marker_name"]
        for charger in chargers:
            if charger["charges_my_cabin"]:
                return charger["marker_name"]
        return None

    def classify_marker(self, marker_name: str, identity: dict[str, Any] | None) -> dict[str, Any] | None:
        """Charger classification for a resolved marker, or None if it isn't a charger."""
        session = self.session_factory()
        try:
            canonical = self._canonical_marker_name(session, marker_name)
            if canonical is None:
                return None
            marker = session.get(MarkerCache, canonical)
            if marker is None or marker.marker_type != CHARGER_MARKER_TYPE:
                return None
            keys = _charger_keys(marker.raw_payload)
            return {"marker_name": canonical, **keys, **_charger_match(keys, identity)}
        finally:
            session.close()

    def _canonical_marker_name(self, session, name: str) -> str | None:
        """Return the stored (canonical) marker name, matching case-insensitively.

        Marker names are sent verbatim to the robot, so the original casing
        must be preserved even when the user/LLM supplies a different case.
        """
        marker = session.get(MarkerCache, name)
        if marker is not None:
            return marker.marker_name
        target = name.strip().lower()
        row = session.scalars(
            select(MarkerCache).where(func.lower(MarkerCache.marker_name) == target)
        ).first()
        return row.marker_name if row is not None else None

    def resolve_location(self, name: str) -> ResolvedLocation:
        normalized = name.strip().lower()
        session = self.session_factory()
        try:
            alias = session.get(LocationAlias, normalized)
            if alias is not None:
                canonical = self._canonical_marker_name(session, alias.marker_name)
                if canonical is not None:
                    return ResolvedLocation(requested_name=name, marker_name=canonical, source="alias")
            canonical = self._canonical_marker_name(session, name)
            if canonical is not None:
                return ResolvedLocation(requested_name=name, marker_name=canonical, source="marker")
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
