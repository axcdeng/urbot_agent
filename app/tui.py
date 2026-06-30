"""Terminal chat console for driving the WATER robot through the AI agent.

Run with the ``alexagent`` command (installed via ``pip install -e .``).

Layout: a scrolling transcript (chat + tool-call trace), a live status bar
(dry-run / e-stop / battery / move state), and an input line.

Key safety behaviour:
* ``Esc``        -> graceful soft stop: let the current movement finish, then
  run no more mission steps (does NOT halt mid-motion).
* ``Esc Esc``    -> ask for confirmation, then a soft EMERGENCY stop
  (``/api/estop?flag=true``) that free-stops the robot immediately.
* Slash commands -> ``/dryrun on|off``, ``/status``, ``/markers``, ``/stop``,
  ``/release``, ``/help``, ``/quit``.

The stop handlers and status polling run off the UI thread, so they respond
even while the model is still generating a reply.

Note: the WATER API only exposes a *soft* e-stop. A true *hardware* hard-stop
is the physical button on the robot and cannot be triggered from software
(see API_EN.md s1.5).
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.prompts import build_completion_prompt
from app.config import get_settings
from app.main import ServiceContainer, build_services
from app.robot.profiles import ROBOT_PROFILES

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# --------------------------------------------------------------------------- #
# UI-agnostic controller (unit-tested in tests/test_tui.py)
# --------------------------------------------------------------------------- #
@dataclass
class CommandResult:
    lines: list[str] = field(default_factory=list)
    quit: bool = False
    clear: bool = False


HELP_LINES = [
    "[b]Commands[/b]",
    "  /dryrun [on|off]   show or toggle dry-run (simulated vs live robot)",
    "  /ip [host[:port]]  show or switch the robot address",
    "  /ip <profile>      switch robot + map by profile (primary, secondary)",
    "  /status            current robot state",
    "  /markers           known markers and aliases",
    "  /stop              graceful stop: finish the current step, run no more",
    "  /release           release the soft e-stop",
    "  /new               start a new chat",
    "  /chats             list previous chats",
    "  /open <n>          re-open a chat from /chats",
    "  /compact [note]    summarize earlier turns to free up context",
    "  /clear             clear the transcript",
    "  /help              this help",
    "  /quit              exit",
    "",
    "[b]Keys[/b]",
    "  Esc                graceful soft stop (finish current step, run no more)",
    "  Esc Esc            confirm, then EMERGENCY stop (immediate free-stop)",
    "  PgUp / PgDn        scroll the transcript",
    "  Shift+↑ / Shift+↓  scroll the transcript one line",
    "",
    "The agent remembers earlier messages in the current chat; /new starts fresh.",
    "Anything not starting with '/' is sent to the AI agent.",
]


class AgentConsole:
    """Thin, synchronous wrapper over the service container.

    Every method here is safe to call from a worker thread; none of them
    touch UI state.
    """

    def __init__(self, services: ServiceContainer):
        self.services = services
        # Each TUI run starts in a fresh chat; previous chats are reachable via
        # /chats + /open. (Not auto-resuming is deliberate — it's why a restart
        # no longer drags an old conversation/mission back into context.)
        self.current_session_id = services.conversation_manager.create_session()
        # Missions dispatched during THIS run — so completion notices only fire
        # for missions the user actually started here (not stale DB history).
        self._session_mission_ids: set[str] = set()

    @property
    def convo(self):
        return self.services.conversation_manager

    # ----- robot / agent actions ----------------------------------------- #
    @property
    def dry_run(self) -> bool:
        return self.services.settings.water_dry_run

    @property
    def live_target(self) -> str:
        return self.services.settings.http_base_url

    @property
    def robot_addr(self) -> str:
        s = self.services.settings
        return f"{s.water_robot_host}:{s.water_http_port}"

    def set_dry_run(self, enabled: bool) -> None:
        # Settings is a shared mutable singleton and WaterRobotClient.dry_run
        # reads it live, so this takes effect on the very next command.
        self.services.settings.water_dry_run = enabled

    def set_robot_host(self, host: str, port: int | None = None) -> None:
        # HttpWaterTransport builds every URL from settings.http_base_url at
        # call time, so changing the host/port here redirects all robot calls
        # (status, move, e-stop, markers) immediately.
        self.services.settings.water_robot_host = host
        if port is not None:
            self.services.settings.water_http_port = port

    def current_profile(self) -> str | None:
        s = self.services.settings
        for name, profile in ROBOT_PROFILES.items():
            if profile.host == s.water_robot_host and profile.port == s.water_http_port:
                return name
        return None

    def use_profile(self, name: str) -> int:
        """Switch to a named profile: its address AND its marker map.

        Returns the number of markers loaded.
        """
        profile = ROBOT_PROFILES[name]
        self.set_robot_host(profile.host, profile.port)
        # Make both the dry-run simulation and the agent's resolution cache use
        # this profile's map.
        self.services.client.set_dry_markers(profile.markers)
        loaded = self.services.location_registry.load_marker_map(profile.markers)
        return len(loaded)

    def chat(self, message: str, on_event=None) -> dict[str, Any]:
        llm = self.services.llm_client
        llm.reset_metrics()
        summary, history = self.convo.build_history(self.current_session_id)
        result = self.services.agent_planner.run_chat(
            message, history=history, summary=summary, on_event=on_event
        )
        self._session_mission_ids.update(result.get("created_mission_ids") or [])
        # Snapshot the answer's metrics BEFORE the title/compaction calls below
        # (they reuse the same LLM client and would otherwise inflate the numbers
        # shown for this turn).
        result["metrics"] = copy.copy(llm.metrics)
        self.convo.record_turn(self.current_session_id, message, result)
        self.convo.maybe_name_session(self.current_session_id)
        if self.convo.should_compact(self.current_session_id):
            report = self.convo.compact(self.current_session_id)
            if report.get("compacted"):
                result["auto_compacted"] = report
        return result

    # ----- chat sessions -------------------------------------------------- #
    def new_session(self) -> str:
        self.current_session_id = self.convo.create_session()
        return self.current_session_id

    def list_chats(self) -> list[dict[str, Any]]:
        return self.convo.list_sessions()

    def open_chat(self, ref: str) -> str | None:
        """Resolve a /chats index (1-based) or session-id prefix to a session."""
        sessions = self.convo.list_sessions()
        target: str | None = None
        if ref.isdigit():
            idx = int(ref) - 1
            if 0 <= idx < len(sessions):
                target = sessions[idx]["session_id"]
        else:
            target = next((s["session_id"] for s in sessions if s["session_id"].startswith(ref)), None)
        if target is not None:
            self.current_session_id = target
        return target

    def render_history_lines(self, session_id: str) -> list[str]:
        info = self.convo.get_session(session_id) or {}
        messages = self.convo.get_messages(session_id)
        title = info.get("title") or session_id[:8]
        lines = [f"[dim]— chat: {title} ({len(messages)} messages) —[/dim]"]
        if info.get("has_summary"):
            lines.append("[dim]— earlier turns are summarized —[/dim]")
        for m in messages:
            if m["role"] == "user":
                lines.append(f"[b cyan]you[/b cyan] {m['content']}")
            else:
                lines.append(f"[b green]agent[/b green] {m['content']}")
                if m.get("actions"):
                    lines.append(f"  [dim]→ {'; '.join(m['actions'])}[/dim]")
        return lines

    def compact_current(self, instructions: str | None = None) -> dict[str, Any]:
        return self.convo.compact(self.current_session_id, instructions)

    def context_fraction(self) -> float:
        return self.convo.context_fraction(self.current_session_id)

    def chat_title(self) -> str | None:
        return (self.convo.get_session(self.current_session_id) or {}).get("title")

    def soft_estop(self) -> dict[str, Any]:
        return self.services.task_manager.emergency_stop()

    def soft_stop(self) -> dict[str, Any]:
        """Graceful stop: let the active mission's current step finish, run no more.

        Only meaningful for a multi-step mission (a lone move just completes). Does
        not touch the robot's in-progress motion — unlike Esc (e-stop) or cancel.
        """
        active = next(
            (m for m in self.services.mission_manager.list_missions()
             if m["status"] in {"created", "running", "waiting"}),
            None,
        )
        if active is None:
            return {"stopped": False, "reason": "no active mission"}
        report = self.services.mission_manager.soft_stop_mission(active["mission_id"])
        report["stopped"] = True
        return report

    def cancel_active_move(self) -> dict[str, Any]:
        return self.services.task_manager.cancel_current_move()

    def release_estop(self) -> dict[str, Any]:
        return self.services.task_manager.release_emergency_stop(True)

    def status(self) -> dict[str, Any]:
        return self.services.state_manager.get_compact_robot_state()

    def markers(self) -> dict[str, Any]:
        return self.services.location_registry.list_locations()

    def poll(self) -> None:
        self.services.task_manager.poll_active_tasks()
        self.services.mission_manager.poll_missions()

    def drain_completions(self) -> list[dict[str, Any]]:
        """Missions that finished since the last check, limited to ones started
        in this run. Each is returned once."""
        finished = self.services.mission_manager.drain_finalized()
        return [m for m in finished if m.get("mission_id") in self._session_mission_ids]

    def summarize_completion(self, mission: dict[str, Any]) -> str:
        """A short, plain 'mission done' line for the user. Asks the model for a
        casual sentence; falls back to a simple template when the LLM is
        off/dry-run (model returns an empty string)."""
        try:
            text = self.services.llm_client.complete(build_completion_prompt(mission), max_tokens=120)
        except Exception:  # noqa: BLE001 - never let a notice crash polling
            text = ""
        text = (text or "").strip()
        if text:
            return text
        name = mission.get("name") or "the mission"
        if mission.get("status") == "succeeded":
            return f"All done — {name} finished."
        if mission.get("status") == "canceled":
            return f"Stopped {name} as you asked."
        problem = mission.get("error_message") or "it ran into a problem"
        return f"{name} didn't finish — {problem}."

    # ----- slash commands ------------------------------------------------- #
    def run_command(self, line: str) -> CommandResult:
        parts = line[1:].strip().split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:]

        if cmd in ("dryrun", "dry"):
            if not args:
                return CommandResult([f"dry-run is {'ON' if self.dry_run else 'OFF'}"])
            value = args[0].lower()
            if value in ("on", "true", "1", "yes"):
                self.set_dry_run(True)
                return CommandResult(["dry-run -> [green]ON[/green] (robot commands are simulated)"])
            if value in ("off", "false", "0", "no"):
                self.set_dry_run(False)
                return CommandResult([f"dry-run -> [red]OFF[/red] — LIVE commands go to {self.live_target}"])
            return CommandResult(["usage: /dryrun on|off"])

        if cmd == "ip":
            profiles = ", ".join(ROBOT_PROFILES)
            if not args:
                here = self.current_profile()
                label = f" [{here}]" if here else ""
                return CommandResult([
                    f"robot address is {self.robot_addr}{label} (target {self.live_target})",
                    f"profiles: {profiles}",
                ])
            name = args[0].lower()
            if name in ROBOT_PROFILES:
                count = self.use_profile(name)
                return CommandResult([
                    f"switched to profile [b]{name}[/b] -> {self.robot_addr}; loaded {count} markers"
                ])
            host, _, port = args[0].partition(":")
            host = host.strip()
            if not host:
                return CommandResult([f"usage: /ip <host>[:port] | <profile> ({profiles})"])
            if port:
                try:
                    self.set_robot_host(host, int(port))
                except ValueError:
                    return CommandResult([f"invalid port: {port!r}"])
            else:
                self.set_robot_host(host)
            return CommandResult([f"robot address -> [b]{self.robot_addr}[/b] (target {self.live_target})"])

        if cmd == "status":
            return CommandResult([_format_status(self.status())])

        if cmd in ("markers", "locations", "ls"):
            data = self.markers()
            names = [m["marker_name"] for m in data.get("markers", [])]
            aliases = [f"{a['alias']} -> {a['marker_name']}" for a in data.get("aliases", [])]
            lines = [f"[b]{len(names)} markers[/b]: " + ", ".join(names)]
            if aliases:
                lines.append(f"[b]{len(aliases)} aliases[/b]: " + ", ".join(aliases))
            return CommandResult(lines)

        if cmd == "release":
            try:
                self.release_estop()
                return CommandResult(["soft e-stop released"])
            except Exception as exc:  # noqa: BLE001 - surface any failure to the user
                return CommandResult([f"could not release e-stop: {exc}"])

        if cmd == "stop":
            report = self.soft_stop()
            if not report.get("stopped"):
                return CommandResult([
                    "no active mission to stop — a single move just finishes on its own. "
                    "(Press Esc for an immediate e-stop.)"
                ])
            tail = " — finishing the current step first" if report.get("finishing_current_step") else ""
            return CommandResult([
                f"soft stop{tail}; canceled {report['canceled_pending_steps']} upcoming step(s). "
                "The robot will not start another step."
            ])

        if cmd == "new":
            self.new_session()
            return CommandResult(["[dim]started a new chat[/dim]"], clear=True)

        if cmd in ("chats", "list", "history"):
            sessions = self.list_chats()
            if not sessions:
                return CommandResult(["no chats yet"])
            lines = ["[b]Chats[/b] (newest first; use /open <n>)"]
            for i, s in enumerate(sessions, 1):
                here = " [green]● current[/green]" if s["session_id"] == self.current_session_id else ""
                title = s["title"] or "(unnamed)"
                lines.append(f"  {i}. {title}  [dim]{s['message_count']} msgs[/dim]{here}")
            return CommandResult(lines)

        if cmd in ("open", "switch"):
            if not args:
                return CommandResult(["usage: /open <number from /chats>"])
            target = self.open_chat(args[0])
            if target is None:
                return CommandResult([f"no such chat: {args[0]} (see /chats)"])
            return CommandResult(self.render_history_lines(target), clear=True)

        if cmd in ("help", "?"):
            return CommandResult(list(HELP_LINES))

        if cmd == "clear":
            return CommandResult(clear=True)

        if cmd in ("quit", "exit", "q"):
            return CommandResult(["bye"], quit=True)

        return CommandResult([f"unknown command: /{cmd} (try /help)"])


def _format_args(args: Any) -> str:
    if isinstance(args, dict):
        return ", ".join(f"{k}={v!r}" for k, v in args.items())
    return str(args)


def _format_status(state: dict[str, Any]) -> str:
    estop = "[red]ESTOP[/red]" if state.get("estop_state") else "ok"
    online = "online" if state.get("online") else "[red]offline[/red]"
    return (
        f"{online} · estop={estop} · battery={state.get('battery_percent')}% · "
        f"move={state.get('move_status')} · target={state.get('current_target') or '-'} · "
        f"floor={state.get('current_floor')}"
    )


# --------------------------------------------------------------------------- #
# Textual application
# --------------------------------------------------------------------------- #
try:  # Import lazily so `import app.tui` doesn't hard-require textual for tests.
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Footer, Header, Input, RichLog, Static

    _TEXTUAL_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without textual
    _TEXTUAL_AVAILABLE = False


if _TEXTUAL_AVAILABLE:

    class ConfirmEstopScreen(ModalScreen[bool]):
        """Confirm before firing the immediate emergency stop (double-Esc)."""

        CSS = """
        ConfirmEstopScreen { align: center middle; background: $background 60%; }
        #estop-confirm { width: 60; height: auto; border: thick $error; background: $surface; padding: 1 2; }
        """

        BINDINGS = [
            Binding("y", "confirm", "E-STOP", priority=True),
            Binding("enter", "confirm", "E-STOP", priority=True),
            Binding("escape", "cancel", "Cancel", priority=True),
            Binding("n", "cancel", "Cancel", priority=True),
        ]

        def compose(self) -> ComposeResult:
            yield Static(
                "[b red]⚠ EMERGENCY STOP[/b red]\n\n"
                "Immediately free-stop the robot?\n\n"
                "[b]Y[/b] / [b]Enter[/b] = e-stop      [b]Esc[/b] / [b]N[/b] = cancel",
                id="estop-confirm",
            )

        def action_confirm(self) -> None:
            self.dismiss(True)

        def action_cancel(self) -> None:
            self.dismiss(False)

    class AlexAgentTUI(App):
        CSS = """
        /* Chat box wraps the transcript AND the live thinking line so the
           spinner reads as the tail of the conversation, not bottom chrome. */
        #chat { height: 1fr; border: round $primary; }
        #log { height: 1fr; padding: 0 1; background: transparent; }
        #thinking { height: auto; color: $text-muted; padding: 0 1; background: transparent; }
        #status { height: 1; background: $panel; color: $text; padding: 0 1; }
        #prompt { dock: bottom; }
        /* Footer + a right-aligned context gauge share the bottom line. */
        #footerbar { dock: bottom; height: 1; }
        #footerbar Footer { dock: none; width: 1fr; height: 1; }
        #ctx { width: auto; height: 1; padding: 0 1; background: $panel; color: $text-muted; content-align: right middle; }
        """

        BINDINGS = [
            Binding("escape", "estop", "Soft stop (2×=E-STOP)", priority=True, show=True),
            Binding("ctrl+c", "quit", "Quit", priority=True, show=True),
            # Scroll the transcript (works even when the input has focus).
            # Plain up/down matters for Terminal.app, whose "Scroll alternate
            # screen" turns the trackpad/wheel into arrow-key input to the app.
            Binding("up", "scroll_chat('up')", "Scroll up", show=True),
            Binding("down", "scroll_chat('down')", "Scroll down", show=True),
            Binding("pageup", "scroll_chat('pageup')", "Page up", show=False),
            Binding("pagedown", "scroll_chat('pagedown')", "Page down", show=False),
            Binding("shift+up", "scroll_chat('up')", "Line up", show=False),
            Binding("shift+down", "scroll_chat('down')", "Line down", show=False),
        ]

        DOUBLE_ESC_WINDOW = 0.45  # seconds

        def __init__(self, console: AgentConsole):
            super().__init__()
            self.ctl = console
            self._last_esc = 0.0
            self._confirming = False
            self._busy = False
            self._spin = 0
            self._think_start = 0.0
            # Start of the CURRENT thinking spell. Reset after each narration /
            # tool line so the live "Thinking…" counter restarts (Claude-Code
            # style); a spell over 30s leaves a grey "Thought for Ns" trace.
            self._segment_start = 0.0
            self._thinking_shown = False
            self._stop = threading.Event()
            # "Mission done" notices queued from the background poll thread,
            # delivered only while idle so they never interleave with a turn.
            self._completion_queue: list[dict[str, Any]] = []
            self._completion_lock = threading.Lock()
            self._delivering_completion = False

        # ----- layout ----------------------------------------------------- #
        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Vertical():
                with Vertical(id="chat"):
                    yield RichLog(id="log", markup=True, wrap=True, highlight=False)
                    yield Static("", id="thinking")
                yield Static("", id="status")
                yield Input(placeholder="Message the agent, or /help …", id="prompt")
            with Horizontal(id="footerbar"):
                yield Footer()
                yield Static("", id="ctx")

        def on_mount(self) -> None:
            self.title = "alex_agent"
            self.log_widget = self.query_one("#log", RichLog)
            self.status_widget = self.query_one("#status", Static)
            self.thinking_widget = self.query_one("#thinking", Static)
            self.ctx_widget = self.query_one("#ctx", Static)
            self._write("[dim]Connected. Type a message or /help. Esc = soft stop · double-Esc = emergency stop.[/dim]")
            mode = "ON (simulated)" if self.ctl.dry_run else f"OFF — LIVE -> {self.ctl.live_target}"
            self._write(f"[dim]dry-run is {mode}[/dim]")
            self.query_one("#prompt", Input).focus()
            self._background_loop()
            self.set_interval(2.0, self._refresh)
            self.set_interval(0.1, self._animate_thinking)

        # ----- input ------------------------------------------------------ #
        def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            event.input.clear()
            if not text:
                return
            if text.startswith("/"):
                body = text[1:].strip()
                cmd = body.split()[0].lower() if body else ""
                # /compact calls the model — run it off the UI thread with the
                # same "thinking" indicator as a chat turn.
                if cmd == "compact":
                    if self._busy:
                        self._write("[yellow]still working on the previous message…[/yellow]")
                        return
                    instructions = body[len("compact"):].strip() or None
                    self._write("[yellow]🗜 Compacting conversation…[/yellow]")
                    self._busy = True
                    self._think_start = self._segment_start = time.monotonic()
                    self._compact_worker(instructions)
                    return
                result = self.ctl.run_command(text)
                if result.clear:
                    self.log_widget.clear()
                for line in result.lines:
                    self._write(line)
                if result.quit:
                    self.exit()
                return
            if self._busy:
                self._write("[yellow]still working on the previous message…[/yellow]")
                return
            self._write(f"[b cyan]you[/b cyan] {text}")
            # Instant acknowledgement so the user gets feedback before the model
            # (which can take several seconds) produces its first output.
            self._write("[b green]agent[/b green] On it.")
            self._busy = True
            self._think_start = self._segment_start = time.monotonic()
            self._chat_worker(text)

        # ----- scrolling --------------------------------------------------- #
        def action_scroll_chat(self, how: str) -> None:
            log = self.log_widget
            if how == "pageup":
                log.scroll_page_up()
            elif how == "pagedown":
                log.scroll_page_down()
            elif how == "up":
                log.scroll_up()
            elif how == "down":
                log.scroll_down()

        # Mouse-wheel anywhere scrolls the transcript (terminals that forward the
        # wheel to the app — the RichLog already handles it directly when hovered).
        def on_mouse_scroll_down(self, event) -> None:
            for _ in range(3):
                self.log_widget.scroll_down(animate=False)
            event.stop()

        def on_mouse_scroll_up(self, event) -> None:
            for _ in range(3):
                self.log_widget.scroll_up(animate=False)
            event.stop()

        # ----- stop actions ----------------------------------------------- #
        def action_estop(self) -> None:
            # Single Esc -> graceful soft stop. Double Esc -> confirm, then a
            # real immediate emergency stop.
            if self._confirming:
                return  # the confirmation modal owns the keyboard
            now = time.monotonic()
            double = (now - self._last_esc) <= self.DOUBLE_ESC_WINDOW
            self._last_esc = now
            if double:
                self._confirming = True
                self.push_screen(ConfirmEstopScreen(), self._on_estop_confirmed)
            else:
                self._soft_stop_worker()

        def _on_estop_confirmed(self, confirmed: bool | None) -> None:
            self._confirming = False
            if confirmed:
                self._estop_worker()
            else:
                self._write("[dim]emergency stop canceled[/dim]")

        # ----- workers (run off the UI thread) ---------------------------- #
        @work(thread=True, group="chat")
        def _chat_worker(self, message: str) -> None:
            def on_event(event: dict[str, Any]) -> None:
                # Hop back to the UI thread to render each event the moment it
                # happens (narration / tool call / tool result).
                self.call_from_thread(self._on_agent_event, event)

            try:
                result = self.ctl.chat(message, on_event=on_event)
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._write, f"[red]chat failed:[/red] {exc}")
            else:
                self.call_from_thread(self._render_result, result)
            finally:
                self.call_from_thread(self._set_idle)

        @work(thread=True, group="chat")
        def _compact_worker(self, instructions: str | None) -> None:
            try:
                report = self.ctl.compact_current(instructions)
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._write, f"[red]compact failed:[/red] {exc}")
            else:
                self.call_from_thread(self._render_compact, report)
            finally:
                self.call_from_thread(self._set_idle)

        @work(thread=True, group="estop")
        def _soft_stop_worker(self) -> None:
            try:
                report = self.ctl.soft_stop()
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._write, f"[red]soft stop error:[/red] {exc}")
                return
            if report.get("stopped"):
                tail = " — finishing the current step" if report.get("finishing_current_step") else ""
                msg = (
                    f"[yellow]⏸ soft stop{tail}: canceled {report['canceled_pending_steps']} "
                    f"upcoming step(s); the robot won't start another.[/yellow]"
                )
            else:
                msg = "[dim]soft stop: nothing running to stop. Double-Esc to emergency-stop.[/dim]"
            self.call_from_thread(self._write, msg)

        @work(thread=True, group="estop")
        def _estop_worker(self) -> None:
            try:
                self.ctl.soft_estop()
                msg = "[b red]🛑 EMERGENCY STOP engaged (soft e-stop)[/b red]"
            except Exception as exc:  # noqa: BLE001
                msg = f"[red]e-stop error:[/red] {exc}"
            self.call_from_thread(
                self._write,
                msg + "  [dim](release with /release; hardware hard-stop is the physical button)[/dim]",
            )

        @work(thread=True, group="poll", exclusive=True)
        def _background_loop(self) -> None:
            # Drives task/mission polling without blocking the event loop.
            # Uses the stop event so quitting returns promptly instead of
            # waiting out a sleep.
            while not self._stop.is_set():
                try:
                    self.ctl.poll()
                    finished = self.ctl.drain_completions()
                    if finished:
                        with self._completion_lock:
                            self._completion_queue.extend(finished)
                    self.call_from_thread(self._maybe_deliver_completion)
                except Exception:  # noqa: BLE001 - polling is best-effort
                    pass
                self._stop.wait(2.0)

        def on_unmount(self) -> None:
            self._stop.set()

        def _maybe_deliver_completion(self) -> None:
            # Deliver one queued "mission done" notice at a time, and only while
            # idle, so it never interleaves with a chat turn the user is running.
            if self._busy or self._delivering_completion:
                return
            with self._completion_lock:
                if not self._completion_queue:
                    return
                mission = self._completion_queue.pop(0)
            self._delivering_completion = True
            self._completion_worker(mission)

        @work(thread=True, group="completion")
        def _completion_worker(self, mission: dict[str, Any]) -> None:
            try:
                text = self.ctl.summarize_completion(mission)
            except Exception as exc:  # noqa: BLE001
                text = f"(couldn't summarize a finished mission: {exc})"
            self.call_from_thread(self._write, f"[b green]agent[/b green] {text}")
            self.call_from_thread(self._completion_done)

        def _completion_done(self) -> None:
            # Drain the next queued notice (if any) now that this one is shown.
            self._delivering_completion = False
            self._maybe_deliver_completion()

        @work(thread=True, group="status", exclusive=True)
        def _refresh(self) -> None:
            try:
                state = self.ctl.status()
            except Exception as exc:  # noqa: BLE001
                state = {"online": False, "move_status": f"err: {exc}"}
            mode = "[green]DRY[/green]" if self.ctl.dry_run else "[red]LIVE[/red]"
            profile = self.ctl.current_profile()
            addr = f"{self.ctl.robot_addr}" + (f" ({profile})" if profile else "")
            try:
                pct = int(round(self.ctl.context_fraction() * 100))
            except Exception:  # noqa: BLE001
                pct = 0
            threshold = int(self.ctl.services.settings.auto_compact_threshold * 100)
            ctx = f"[yellow]ctx {pct}%[/yellow]" if pct >= threshold else f"[dim]ctx {pct}%[/dim]"
            title = self.ctl.chat_title() or "new chat"
            self.call_from_thread(
                self.status_widget.update,
                f"{mode} {addr} · {_format_status(state)}",
            )
            self.call_from_thread(self.ctx_widget.update, ctx)
            self.call_from_thread(self._set_subtitle, title)

        def _set_subtitle(self, title: str) -> None:
            self.sub_title = title

        def _animate_thinking(self) -> None:
            if self._busy:
                frame = SPINNER[self._spin % len(SPINNER)]
                self._spin += 1
                # Count from the current spell, which restarts after each
                # narration / tool line (so the timer reflects "since last
                # output", Claude-Code style).
                elapsed = time.monotonic() - self._segment_start
                self.thinking_widget.update(f"{frame} ✻ Thinking… ({elapsed:.0f}s)  [dim]· Esc soft-stop[/dim]")
                self._thinking_shown = True
            elif self._thinking_shown:
                self.thinking_widget.update("")
                self._thinking_shown = False

        # ----- rendering helpers ------------------------------------------ #
        def _write(self, renderable: Any) -> None:
            self.log_widget.write(renderable)

        def _reset_think_segment(self) -> None:
            # End the current thinking spell. If it ran long, leave a permanent
            # grey trace in the transcript; then start a fresh spell so the live
            # counter restarts from 0 for whatever the agent does next.
            now = time.monotonic()
            elapsed = now - self._segment_start
            if self._busy and elapsed > 30:
                self._write(f"[dim]Thought for {elapsed:.0f}s[/dim]")
            self._segment_start = now

        def _set_idle(self) -> None:
            # Close out the final spell (may leave a "Thought for Ns" trace),
            # then clear the live indicator.
            self._reset_think_segment()
            self._busy = False
            self.thinking_widget.update("")
            self._thinking_shown = False
            # A mission may have finished while the user was mid-turn; now idle,
            # deliver any queued completion notice.
            self._maybe_deliver_completion()

        def _on_agent_event(self, event: dict[str, Any]) -> None:
            # Render a single streamed event the moment it happens, then restart
            # the thinking spell so the live counter resets for the next step.
            kind = event.get("type")
            if kind == "narration":
                text = event.get("text", "")
                if text:
                    self._write(f"[b green]agent[/b green] {text}")
            elif kind == "tool_call":
                self._write_tool_call(event.get("name"), event.get("arguments"))
            elif kind == "tool_result":
                self._write_tool_detail(event.get("payload"))
            self._reset_think_segment()

        def _write_tool_call(self, name: Any, args: Any) -> None:
            self._write(f"  [magenta]🔧 {name}[/magenta]([cyan]{_format_args(args)}[/cyan])")

        def _write_tool_detail(self, payload: Any) -> None:
            payload = payload or {}
            outcome = payload.get("error_message") or payload.get("status")
            target = payload.get("requested_target") or payload.get("marker_name")
            task_id = payload.get("task_id")
            detail = " · ".join(
                p for p in (
                    f"target={target}" if target else "",
                    f"task={task_id[:8]}" if task_id else "",
                    str(outcome) if outcome else "",
                ) if p
            )
            if detail:
                self._write(f"     [dim]→ {detail}[/dim]")

        def _render_compact(self, report: dict[str, Any]) -> None:
            if report.get("compacted"):
                was = int(round(report.get("before_fraction", 0) * 100))
                self._write(
                    f"[yellow]🗜 Compacted: summarized {report.get('removed', 0)} earlier "
                    f"messages, kept the last {report.get('kept', 0)} (context was {was}% full)[/yellow]"
                )
            else:
                self._write(f"[dim]nothing to compact ({report.get('reason', 'n/a')})[/dim]")

        def _render_result(self, result: dict[str, Any]) -> None:
            # When the turn was streamed, its tool calls / details were already
            # rendered live by _on_agent_event — only finalize here (final reply,
            # compaction notice, metrics). The non-streamed path renders the
            # whole batch.
            if not result.get("streamed"):
                for call in result.get("tool_calls", []):
                    self._write_tool_call(call.get("name"), call.get("arguments"))
                    self._write_tool_detail(call.get("payload"))
                for mission_id in result.get("created_mission_ids", []):
                    self._write(f"  [blue]＋ mission {mission_id[:8]}[/blue]")
            response = result.get("assistant_response", "")
            if response:
                self._write(f"[b green]agent[/b green] {response}")
            auto = result.get("auto_compacted")
            if auto and auto.get("compacted"):
                was = int(round(auto.get("before_fraction", 0) * 100))
                self._write(
                    f"[yellow]🗜 Auto-compacted: summarized {auto.get('removed', 0)} earlier "
                    f"messages (context was {was}% full)[/yellow]"
                )
            metrics = result.get("metrics")
            if metrics is not None and getattr(metrics, "calls", 0):
                self._write(
                    f"[dim]↑{metrics.prompt_tokens} in · ↓{metrics.completion_tokens} out · "
                    f"{metrics.tps:.0f} tok/s · {metrics.elapsed_s:.1f}s · {metrics.calls} call(s)[/dim]"
                )


def _configure_logging() -> None:
    # Textual owns the terminal; library/app logging to the console corrupts the
    # UI. Route everything to a file and quiet the chatty HTTP loggers.
    logging.basicConfig(
        filename="alexagent.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> None:
    if not _TEXTUAL_AVAILABLE:
        raise SystemExit(
            "The TUI requires the 'textual' package. Install it with: pip install -e ."
        )
    _configure_logging()
    services = build_services(get_settings())
    AlexAgentTUI(AgentConsole(services)).run()


if __name__ == "__main__":
    main()
