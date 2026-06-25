# alex_agent

`alex_agent` is a local FastAPI orchestrator for one WATER chassis robot. The backend owns safety validation, marker resolution, task lifecycle tracking, and translation from safe internal actions to documented WATER API requests. A local `llama.cpp` OpenAI-compatible server can optionally turn natural-language chat into those safe internal tool calls.

This version also supports durable multi-step missions. A mission is a stored workflow made of ordered steps like move, wait, move again, and return to charger. Missions continue in the background through the polling loop, and the backend automatically dispatches the next step when the prior one succeeds.

## Stack

- Python 3.11+
- FastAPI + Uvicorn
- SQLite + SQLAlchemy
- `httpx` for WATER and LLM HTTP calls
- Jinja2 + vanilla JS for a lightweight dashboard

## Project Layout

- `app/main.py`: FastAPI entrypoint and service wiring
- `app/water/`: WATER API client, schemas, and normalization
- `app/robot/`: location registry, safety checks, state manager, and task manager
- `app/agent/`: LLM client, prompt rules, tool registry, mission planner, and bounded planner loop
- `app/robot/mission_manager.py`: durable mission execution, step queueing, and compact summaries
- `app/api/`: route modules and request schemas
- `scripts/`: smoke and marker-sync helpers
- `tests/`: unit and integration coverage

## Setup

1. Create and activate a virtual environment.
2. Install the project:

```bash
pip install -e .[dev]
```

3. Copy `.env.example` to `.env` and adjust values as needed.

## Running In Dry-Run Mode

Dry-run mode is enabled by default with `WATER_DRY_RUN=true`.

```bash
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the dashboard.

In dry-run mode:

- the robot reports as online and idle
- markers include `charging`, `room_205`, `kitchen_pickup`, and `front_desk`
- move requests create tasks and simulate progress
- no real WATER network calls are made

## Connecting To A Real Robot

Update `.env`:

```bash
WATER_DRY_RUN=false
WATER_ROBOT_HOST=192.168.10.10
WATER_HTTP_PORT=9001
WATER_TCP_PORT=31001
```

The v1 app uses documented HTTP endpoints for:

- `/api/robot_status`
- `/api/robot_info`
- `/api/get_power_status`
- `/api/get_battery_status`
- `/api/get_current_location`
- `/api/markers/query_list`
- `/api/markers/query_brief`
- `/api/map/get_current_map`
- `/api/move`
- `/api/move/cancel`
- `/api/estop`

TCP realtime subscriptions are intentionally stubbed for later work.

## Running The LLM

Start your local `llama.cpp` OpenAI-compatible server separately, for example:

```bash
./server -m /path/to/qwen-model.gguf --host 127.0.0.1 --port 8080
```

Set in `.env`:

```bash
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_MODEL=qwen
LLM_API_KEY=local-llama
LLM_ENABLED=true
```

If tool calling is unavailable or unreliable, `/agent/chat` falls back to a strict JSON planner prompt.

## API Examples

Check robot status:

```bash
curl http://127.0.0.1:8000/robot/status
```

List markers:

```bash
curl http://127.0.0.1:8000/robot/markers
```

Move to a marker:

```bash
curl -X POST http://127.0.0.1:8000/tasks/move-marker \
  -H "Content-Type: application/json" \
  -d '{"marker_name":"room_205"}'
```

Cancel movement:

```bash
curl -X POST http://127.0.0.1:8000/tasks/cancel
```

Natural-language chat:

```bash
curl -X POST http://127.0.0.1:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Go to room 205"}'
```

Create a durable mission directly:

```bash
curl -X POST http://127.0.0.1:8000/missions \
  -H "Content-Type: application/json" \
  -d '{
    "user_request":"Go to room 205, wait 30 seconds, then return to charger",
    "steps":[
      {"step_type":"move_marker","marker_name":"room_205"},
      {"step_type":"wait","wait_seconds":30},
      {"step_type":"return_to_charger"}
    ],
    "auto_replan":true
  }'
```

Ask the LLM to plan a mission from natural language:

```bash
curl -X POST http://127.0.0.1:8000/missions/plan \
  -H "Content-Type: application/json" \
  -d '{"message":"Go to room 205, wait 30 seconds, then go to kitchen pickup and finally return to charger"}'
```

## Helper Scripts

Connectivity smoke test:

```bash
python scripts/smoke_test_robot.py
```

Sync markers into SQLite:

```bash
python scripts/sync_markers.py
```

## Safety Notes

- The LLM never calls raw WATER endpoints directly.
- Direct velocity control and joystick commands are intentionally excluded.
- Movement is blocked when the robot is offline, estopped, below the minimum battery threshold, already moving without interruption permission, targeting an unknown marker, or reporting an error state.
- Emergency stop is always available.
- Releasing emergency stop requires explicit confirmation.
- Long-running missions are compacted into short summaries before replanning so smaller-context local models only see the mission goal, recent completed steps, current step, next steps, and latest error instead of the full raw history.

## Testing

```bash
pytest
```

## Intentionally Stubbed In V1

- TCP realtime callback handling
- charger docking via `/api/docking`
- multi-robot routing
- direct velocity control
