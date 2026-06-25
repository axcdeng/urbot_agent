from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import MoveLocationRequest, MoveMarkerRequest, ReleaseEstopRequest


router = APIRouter(prefix="/tasks", tags=["tasks"])


def _handle_error(exc: Exception):
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/move-marker")
def move_marker(body: MoveMarkerRequest, request: Request):
    try:
        return request.app.state.services.task_manager.create_move_marker_task(body.marker_name, body.allow_interruption)
    except Exception as exc:
        _handle_error(exc)


@router.post("/move-location")
def move_location(body: MoveLocationRequest, request: Request):
    try:
        return request.app.state.services.task_manager.create_move_location_task(body.x, body.y, body.theta, body.allow_interruption)
    except Exception as exc:
        _handle_error(exc)


@router.post("/cancel")
def cancel_task(request: Request):
    try:
        return request.app.state.services.task_manager.cancel_current_move()
    except Exception as exc:
        _handle_error(exc)


@router.post("/return-to-charger")
def return_to_charger(request: Request):
    try:
        return request.app.state.services.task_manager.return_to_charger()
    except Exception as exc:
        _handle_error(exc)


@router.post("/emergency-stop")
def emergency_stop(request: Request):
    try:
        return request.app.state.services.task_manager.emergency_stop()
    except Exception as exc:
        _handle_error(exc)


@router.post("/release-emergency-stop")
def release_emergency_stop(body: ReleaseEstopRequest, request: Request):
    try:
        return request.app.state.services.task_manager.release_emergency_stop(body.confirmed)
    except Exception as exc:
        _handle_error(exc)


@router.get("")
def list_tasks(request: Request):
    return request.app.state.services.task_manager.list_tasks()


@router.get("/{task_id}")
def get_task(task_id: str, request: Request):
    try:
        return request.app.state.services.task_manager.get_task(task_id)
    except Exception as exc:
        _handle_error(exc)
