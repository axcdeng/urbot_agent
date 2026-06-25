from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter(prefix="/robot", tags=["robot"])


@router.get("/status")
def get_robot_status(request: Request):
    return request.app.state.services.state_manager.get_robot_state().to_dict()


@router.get("/info")
def get_robot_info(request: Request):
    return request.app.state.services.state_manager.get_robot_info()


@router.get("/battery")
def get_robot_battery(request: Request):
    return request.app.state.services.state_manager.get_battery_status()


@router.get("/location")
def get_robot_location(request: Request):
    return request.app.state.services.state_manager.get_robot_location()


@router.get("/map")
def get_robot_map(request: Request):
    return request.app.state.services.state_manager.get_robot_map()


@router.get("/markers")
def get_robot_markers(request: Request):
    return request.app.state.services.location_registry.list_locations()["markers"]
