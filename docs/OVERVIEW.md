# alex_agent — Project Overview

`alex_agent` is a local orchestrator for a single **WATER** chassis robot. A
natural-language **AI agent** (a local LLM) turns chat into a small set of
**safe, validated** robot actions; the backend owns safety, marker resolution,
task lifecycle, and translation to the documented WATER HTTP API.

It ships two front ends over the same service layer:

- a **FastAPI** HTTP server (`app/main.py`) — programmatic API + optional web UI
- a **Textual terminal UI** (`app/tui.py`, command `alexagent`) — see
  [TUI.md](TUI.md)

The robot is never driven by raw velocity/joystick commands or raw WATER calls
from the model — only through tools the backend validates.

---

## Stack

- Python 3.11+
- FastAPI + Uvicorn (HTTP API), Textual (TUI)
- SQLite + SQLAlchemy (markers, aliases, tasks, missions, events)
- `httpx` for WATER and LLM HTTP calls
- A local **OpenAI-compatible** LLM server (`mlx_lm.server` serving
  `mlx-community/Qwen3-32B-4bit` on the Mac Studio)

---

## How a chat turn flows

```
user message
   │
   ▼
AgentPlanner.run_chat                      (app/agent/planner.py)
   │   multi-step request ("…then…")? ──► MissionPlanner → durable mission
   │
   ▼ otherwise: tool-calling loop (≤4 rounds)
LLMClient.chat_with_tools  ──►  local LLM (/v1/chat/completions)
   │        returns assistant text + OpenAI tool_calls
   ▼
AgentToolRegistry.execute  ──►  TaskManager / StateManager / MissionManager
   │        e.g. move_to_location, return_to_charger, emergency_stop
   ▼
SafetyValidator gate ──► WaterRobotClient ──► robot (or dry-run simulation)
```

If tool calling fails, the planner falls back to a strict **JSON planner**
prompt (`LLMClient.json_plan`) that returns a single `{response, action}` object.

---

## Components

| Area | File | Responsibility |
|------|------|----------------|
| Config | `app/config.py` | `Settings` (env-driven): robot host/port, dry-run, LLM url/model/timeouts, limits |
| WATER client | `app/water/client.py` | HTTP transport + **dry-run simulation**; swappable `dry_markers` |
| Normalizer | `app/water/normalizer.py` | Normalize robot status / marker payloads; quaternion→theta |
| State | `app/robot/state_manager.py` | `get_robot_state()` — resilient to optional-endpoint failures |
| Locations | `app/robot/locations.py` | Marker cache, aliases, **case-insensitive resolution**, `load_marker_map` |
| Safety | `app/robot/safety.py` | Gates moves (offline / e-stop / low battery / error / already moving / unknown target) |
| Tasks | `app/robot/task_manager.py` | Move/cancel/e-stop task lifecycle + polling |
| Missions | `app/robot/mission_manager.py` | Durable multi-step missions, step queueing, replanning, compact summaries |
| LLM client | `app/agent/llm_client.py` | Chat + tool calling, JSON planning, `<think>`/fence-tolerant parsing, `LLMMetrics` |
| Tools | `app/agent/tools.py` | Tool definitions + execution (`AgentToolRegistry`) |
| Prompts | `app/agent/prompts.py` | System rules + JSON planner prompts |
| Planner | `app/agent/planner.py` | `run_chat` tool loop + JSON fallback |
| Mission planner | `app/agent/mission_planner.py` | Detect multi-step intent; plan/replan steps |
| Profiles | `app/robot/profiles.py` | Named robot profiles (address + map): `primary`, `secondary` |
| App wiring | `app/main.py` | `build_services()` → `ServiceContainer`; FastAPI app |
| TUI | `app/tui.py` | Terminal chat console (`alexagent`) |

---

## Markers, aliases, and resolution

- Markers come from the robot's `/api/markers/query_list` (live) or from
  `DRY_MARKERS` (dry-run). Both use the same dict-keyed shape
  (`results[name] = {floor, key, pose:{position, orientation}}`).
- The deployed **1F map** is loaded as `DRY_MARKERS` in `app/water/client.py`.
- Marker names can be mixed-case / non-ASCII (e.g. `Meetingroom`,
  `扫地机维护点_1F_1`). `LocationRegistry.resolve_location` matches
  **case-insensitively** but returns the **canonical** name that is sent to the
  robot. Aliases (e.g. `charger → charge_point_1F_1`) are verified to point at a
  marker that exists.

---

## Safety model

- **Dry-run by default** (`WATER_DRY_RUN=true`): all robot I/O is simulated;
  the real robot is never contacted. Set `WATER_DRY_RUN=false` for live.
- Moves are blocked when the robot is offline, in e-stop, below the battery
  threshold, already moving (without interruption permission), targeting an
  unknown marker, or reporting an error.
- **E-stop**: the WATER API only exposes a **soft** e-stop
  (`/api/estop?flag=true|false`). A hardware **hard** stop is the physical
  button and cannot be triggered from software.
- Releasing e-stop requires explicit confirmation.

---

## Configuration (`.env`, see `.env.example`)

| Key | Default | Notes |
|-----|---------|-------|
| `WATER_ROBOT_HOST` | `10.1.17.225` | Robot HTTP host |
| `WATER_HTTP_PORT` | `9001` | Robot HTTP port |
| `WATER_DRY_RUN` | `true` | Simulate robot I/O |
| `MIN_MOVE_BATTERY_PERCENT` | `20` | Move battery floor |
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible server |
| `LLM_MODEL` | `mlx-community/Qwen3-32B-4bit` | **Exact** id from `GET /v1/models` |
| `LLM_ENABLED` | `true` | Turn the agent on/off |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-call timeout |
| `LLM_MAX_TOKENS` | `1024` | Per-call output cap |

> `mlx_lm.server` treats the `model` field as a Hugging Face repo id and
> returns **HTTP 404** for an unknown value (e.g. a bare `qwen`), so `LLM_MODEL`
> must match exactly.

---

## HTTP API (selected)

Served by `uvicorn app.main:app`. Prefixes: `/robot`, `/tasks`, `/agent`,
`/locations`, `/missions`, plus `/health`.

- `GET /health`
- `GET /robot/status` · `/robot/info` · `/robot/battery` · `/robot/location` · `/robot/map` · `/robot/markers`
- `POST /tasks/move-marker` · `/tasks/move-location` · `/tasks/cancel` · `/tasks/return-to-charger` · `/tasks/emergency-stop` · `/tasks/release-emergency-stop` · `GET /tasks/{task_id}`
- `POST /agent/chat` (chat → tools) · `POST /agent/plan` (plan a mission)
- `GET /locations` · `POST /locations/alias` · `DELETE /locations/alias/{alias}`
- `POST /missions` · `GET /missions/{id}` · `GET /missions/{id}/context` · `POST /missions/{id}/cancel`

---

## Running

```bash
conda activate mlx           # the Mac Studio LLM env
pip install -e .             # installs deps + the `alexagent` command

# Terminal UI (recommended for interactive use)
alexagent

# or the HTTP server
uvicorn app.main:app --reload   # http://127.0.0.1:8000
```

The LLM must be reachable at `LLM_BASE_URL` (on the Mac Studio,
`mlx_lm.server` runs on `:8080`).

---

## Tests

```bash
pytest
```

Tests run entirely in dry-run with temporary SQLite DBs — they never touch the
real robot or the LLM server.
