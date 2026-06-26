from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    water_robot_host: str = "10.1.17.225"
    water_http_port: int = 9001
    water_tcp_port: int = 31001
    water_timeout_seconds: float = 10.0
    water_dry_run: bool = True
    min_move_battery_percent: int = 20

    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = "local-llama"
    llm_model: str = "mlx-community/Qwen3-32B-4bit"
    llm_enabled: bool = True
    llm_dry_run: bool = False
    llm_timeout_seconds: float = 120.0
    llm_max_tokens: int = 1024

    mission_poll_seconds: float = 2.0
    mission_max_replans: int = 2
    agent_max_recent_events: int = 6
    agent_max_location_names: int = 40
    agent_max_aliases: int = 25

    database_url: str = Field(default_factory=lambda: f"sqlite:///{PROJECT_ROOT / 'alex_agent.db'}")

    @property
    def http_base_url(self) -> str:
        return f"http://{self.water_robot_host}:{self.water_http_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
