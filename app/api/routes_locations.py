from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import LocationAliasRequest


router = APIRouter(prefix="/locations", tags=["locations"])


@router.get("")
def list_locations(request: Request):
    return request.app.state.services.location_registry.list_locations()


@router.post("/alias")
def add_alias(body: LocationAliasRequest, request: Request):
    try:
        return request.app.state.services.location_registry.add_alias(body.alias, body.marker_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/alias/{alias}")
def delete_alias(alias: str, request: Request):
    try:
        request.app.state.services.location_registry.delete_alias(alias)
        return {"deleted": alias}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
