# alex_agent TUI (`alexagent`)

A terminal chat console for talking to the AI agent and driving the WATER robot.
It runs in-process against the same services as the HTTP API, so it shares the
dry-run setting, marker map, safety checks, and mission engine.

See [OVERVIEW.md](OVERVIEW.md) for the system as a whole.

---

## Launch

```bash
conda activate mlx
pip install -e .     # first time, or after dependency/entry-point changes
alexagent
```

On startup it builds the services, loads the active marker map, and connects to
the LLM at `LLM_BASE_URL`. **Dry-run is on by default** — nothing physical
happens until you run `/dryrun off`.

---

## Layout

```
┌──────────────────────────────────────────────┐
│ transcript: your messages, 🔧 tool calls,     │  ← scrolls
│ agent replies, and a dim ↑/↓ token line       │
├──────────────────────────────────────────────┤
│ ⠹ Thinking… (3s)  · Esc to e-stop             │  ← only while the model works
├──────────────────────────────────────────────┤
│ DRY 10.1.17.225:9001 (primary) · online ·     │  ← live status bar
│ estop=ok · battery=82% · move=idle · …        │
├──────────────────────────────────────────────┤
│ > message the agent, or /help …               │  ← input
└──────────────────────────────────────────────┘
```

- **Transcript** — your input (`you`), each tool call as
  `🔧 name(args)` with a `→ target/task/outcome` line, the agent reply
  (`agent`), and a usage line.
- **Thinking line** — an animated spinner with elapsed seconds shown only while
  a reply is being generated; it disappears when the turn finishes.
- **Status bar** — `DRY|LIVE  host:port (profile) · online · estop · battery ·
  move · target · floor`, refreshed every ~2s.
- **Input** — plain text is sent to the agent; `/`-prefixed text is a command.

---

## Keys

| Key | Action |
|-----|--------|
| `Esc` | **Soft e-stop**, immediately (`/api/estop?flag=true`) |
| `Esc` `Esc` (within ~0.45s) | **PANIC**: soft e-stop **+ cancel** the active move |
| `Ctrl+C` | Quit |
| `Enter` | Send the message / run the command |

E-stop runs on a worker thread, so it fires **instantly even while the model is
still generating** — the in-flight reply can't beat it to the robot. Note: a
true *hardware* hard-stop is the physical button and can't be triggered from
software; double-Esc is the strongest software action available.

Release a soft e-stop with `/release`.

---

## Commands

| Command | Description |
|---------|-------------|
| `/dryrun` | Show whether dry-run is on |
| `/dryrun on` / `/dryrun off` | Toggle simulation vs **live** robot commands |
| `/ip` | Show the current robot address + profile + available profiles |
| `/ip <host>` / `/ip <host>:<port>` | Switch the robot address (map unchanged) |
| `/ip <profile>` | Switch **address and map** by profile (`primary`, `secondary`) |
| `/status` | Print the current robot state |
| `/markers` | List known markers and aliases |
| `/release` | Release the soft e-stop |
| `/clear` | Clear the transcript |
| `/help` | Show help |
| `/quit` | Exit |

Anything not starting with `/` is sent to the AI agent.

### Switching robots — profiles

Profiles live in `app/robot/profiles.py` and bundle an address with a marker
map:

- `primary`  → `10.1.17.225:9001`, the 1F map
- `secondary` → `10.1.16.160:9001`, its map (currently the same 1F map; point
  `SECONDARY_MARKERS` at a different dict to diverge)

`/ip secondary` switches the endpoint **and** reloads that profile's map for
both the agent's resolution and the dry-run simulator; `/ip primary` switches
back. Because every robot request is built from the live setting, the change
takes effect on the very next command — no restart.

---

## Reading the output

A completed turn looks like:

```
you  go to the meeting room
  🔧 move_to_location(location_name='meeting room')
     → target=meeting room · task=4f9c1a2b · OK
agent  Heading to the meeting room now.
↑55 in · ↓150 out · 30 tok/s · 5.0s · 2 call(s)
```

- The **🔧 lines** document each tool the agent invoked and its outcome.
- The **dim line** is per-turn LLM usage: input tokens, output tokens, tokens/s
  (output ÷ time spent in the model), wall-clock seconds, and how many model
  calls the turn took (the tool loop can make several). It only appears when a
  real LLM call happened (not in dry/disabled LLM mode).

---

## Dry-run vs live

- **Dry-run (default)** — robot actions are simulated in memory; safe for
  testing prompts, tool calls, and missions.
- **Live** (`/dryrun off`) — commands go to the configured robot
  (`host:port`). The status bar shows `LIVE` in red. Use e-stop (`Esc`) freely;
  it is always available.

---

## Logging

To keep the UI clean, all logging is redirected to **`alexagent.log`** in the
working directory (and `httpx` is quieted). Tail it for debugging:

```bash
tail -f alexagent.log
```

---

## How it stays responsive

The TUI runs blocking work off the UI thread:

- **chat** runs in a worker thread (the model can take seconds);
- **task/mission polling** runs in a background thread loop;
- **status refresh** runs in a worker thread;
- the **e-stop** handler dispatches its own worker.

So typing, the Thinking animation, and especially **Esc** stay responsive no
matter what the model or robot is doing.
