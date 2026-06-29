from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent.llm_client import LLMClient
from app.agent.mission_planner import MissionPlanner
from app.agent.planner import AgentPlanner
from app.agent.tools import AgentToolRegistry
from app.api.routes_agent import router as agent_router
from app.api.routes_locations import router as locations_router
from app.api.routes_missions import router as missions_router
from app.api.routes_robot import router as robot_router
from app.api.routes_tasks import router as tasks_router
from app.config import PROJECT_ROOT, Settings, get_settings
from app.db import create_session_factory, init_db
from app.robot.locations import LocationRegistry
from app.robot.mission_manager import MissionManager
from app.robot.safety import SafetyValidator
from app.robot.state_manager import StateManager
from app.robot.task_manager import TaskManager
from app.water.client import WaterRobotClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))


@dataclass
class ServiceContainer:
    settings: Settings
    client: WaterRobotClient
    state_manager: StateManager
    location_registry: LocationRegistry
    safety: SafetyValidator
    task_manager: TaskManager
    mission_manager: MissionManager
    llm_client: LLMClient
    mission_planner: MissionPlanner
    agent_tools: AgentToolRegistry
    agent_planner: AgentPlanner


def build_services(settings: Settings) -> ServiceContainer:
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    client = WaterRobotClient(settings)
    state_manager = StateManager(client)
    location_registry = LocationRegistry(session_factory, client)
    location_registry.sync_markers()
    safety = SafetyValidator(settings)
    task_manager = TaskManager(session_factory, client, state_manager, location_registry, safety)
    llm_client = LLMClient(settings)
    mission_planner = MissionPlanner(llm_client, location_registry, state_manager, settings)
    mission_manager = MissionManager(session_factory, task_manager, state_manager, location_registry, settings, mission_planner=mission_planner)
    agent_tools = AgentToolRegistry(state_manager, location_registry, task_manager, mission_manager, mission_planner)
    agent_planner = AgentPlanner(llm_client, agent_tools, state_manager, location_registry, mission_manager, mission_planner)
    return ServiceContainer(
        settings=settings,
        client=client,
        state_manager=state_manager,
        location_registry=location_registry,
        safety=safety,
        task_manager=task_manager,
        mission_manager=mission_manager,
        llm_client=llm_client,
        mission_planner=mission_planner,
        agent_tools=agent_tools,
        agent_planner=agent_planner,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = build_services(app_settings)
        app.state.services = services
        await services.task_manager.start_polling()
        await services.mission_manager.start_polling()
        try:
            yield
        finally:
            await services.mission_manager.stop_polling()
            await services.task_manager.stop_polling()

    app = FastAPI(title="alex_agent", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")), name="static")
    app.include_router(robot_router)
    app.include_router(tasks_router)
    app.include_router(agent_router)
    app.include_router(locations_router)
    app.include_router(missions_router)

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "dry_run": app.state.services.settings.water_dry_run,
            "llm_enabled": app.state.services.settings.llm_enabled,
            "missions_enabled": True,
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return templates.TemplateResponse("dashboard.html", {"request": request})

    return app


app = create_app()
