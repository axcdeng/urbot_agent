from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import AgentChatRequest


router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat")
def chat_with_agent(body: AgentChatRequest, request: Request):
    try:
        return request.app.state.services.agent_planner.run_chat(body.message)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
