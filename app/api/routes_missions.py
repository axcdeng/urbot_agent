from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import MissionCreateRequest, MissionPlanRequest


router = APIRouter(prefix="/missions", tags=["missions"])


@router.post("")
def create_mission(body: MissionCreateRequest, request: Request):
    try:
        steps = [step.model_dump() for step in body.steps]
        return request.app.state.services.mission_manager.create_mission(
            user_request=body.user_request,
            steps=steps,
            name=body.name,
            auto_replan=body.auto_replan,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/plan")
def plan_mission(body: MissionPlanRequest, request: Request):
    try:
        planned = request.app.state.services.mission_planner.plan_steps_from_text(body.message)
        mission = request.app.state.services.mission_manager.create_mission(
            user_request=body.message,
            steps=planned["steps"],
            name=planned.get("mission_name"),
            auto_replan=body.auto_replan,
        )
        return {"planner_response": planned.get("response"), "mission": mission}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def list_missions(request: Request):
    return request.app.state.services.mission_manager.list_missions()


@router.get("/{mission_id}")
def get_mission(mission_id: str, request: Request):
    try:
        return request.app.state.services.mission_manager.get_mission(mission_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{mission_id}/context")
def get_mission_context(mission_id: str, request: Request):
    try:
        return request.app.state.services.mission_manager.get_mission_context(mission_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{mission_id}/cancel")
def cancel_mission(mission_id: str, request: Request):
    try:
        return request.app.state.services.mission_manager.cancel_mission(mission_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
