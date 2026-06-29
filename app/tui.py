"""Terminal chat console for driving the WATER robot through the AI agent.

Run with the ``alexagent`` command (installed via ``pip install -e .``).

Layout: a scrolling transcript (chat + tool-call trace), a live status bar
(dry-run / e-stop / battery / move state), and an input line.

Key safety behaviour:
* ``Esc``        -> soft e-stop (``/api/estop?flag=true``) immediately.
* ``Esc Esc``    -> PANIC: soft e-stop + cancel the active move.
* Slash commands -> ``/dryrun on|off``, ``/status``, ``/markers``,
  ``/release``, ``/help``, ``/quit``.

The e-stop handler and status polling run off the UI thread, so e-stop fires
instantly even while the model is still generating a reply.

Note: the WATER API only exposes a *soft* e-stop. A true *hardware* hard-stop
is the physical button on the robot and cannot be triggered from software
(see API_EN.md s1.5); double-Esc therefore engages the strongest software
action available (soft e-stop + cancel).
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

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
    "  Esc                soft e-stop (immediate)",
    "  Esc Esc            PANIC: soft e-stop + cancel active move",
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

    def chat(self, message: str) -> dict[str, Any]:
        llm = self.services.llm_client
        llm.reset_metrics()
        summary, history = self.convo.build_history(self.current_session_id)
        result = self.services.agent_planner.run_chat(message, history=history, summary=summary)
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
    from textual.containers import Vertical
    from textual.widgets import Footer, Header, Input, RichLog, Static

    _TEXTUAL_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without textual
    _TEXTUAL_AVAILABLE = False


if _TEXTUAL_AVAILABLE:

    class AlexAgentTUI(App):
        CSS = """
        #log { height: 1fr; border: round $primary; padding: 0 1; }
        #thinking { height: 1; color: $warning; padding: 0 1; }
        #status { height: 1; background: $panel; color: $text; padding: 0 1; }
        #prompt { dock: bottom; }
        """

        BINDINGS = [
            Binding("escape", "estop", "E-STOP", priority=True, show=True),
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
            self._busy = False
            self._spin = 0
            self._think_start = 0.0
            self._thinking_shown = False
            self._stop = threading.Event()

        # ----- layout ----------------------------------------------------- #
        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Vertical():
                yield RichLog(id="log", markup=True, wrap=True, highlight=False)
                yield Static("", id="thinking")
                yield Static("", id="status")
                yield Input(placeholder="Message the agent, or /help …", id="prompt")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "alex_agent"
            self.log_widget = self.query_one("#log", RichLog)
            self.status_widget = self.query_one("#status", Static)
            self.thinking_widget = self.query_one("#thinking", Static)
            self._write("[dim]Connected. Type a message or /help. Esc = soft e-stop.[/dim]")
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
                    self._think_start = time.monotonic()
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
            self._busy = True
            self._think_start = time.monotonic()
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

        # ----- e-stop ----------------------------------------------------- #
        def action_estop(self) -> None:
            now = time.monotonic()
            panic = (now - self._last_esc) <= self.DOUBLE_ESC_WINDOW
            self._last_esc = now
            self._estop_worker(panic)

        # ----- workers (run off the UI thread) ---------------------------- #
        @work(thread=True, group="chat")
        def _chat_worker(self, message: str) -> None:
            try:
                result = self.ctl.chat(message)
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
        def _estop_worker(self, panic: bool) -> None:
            try:
                self.ctl.soft_estop()
                msg = "[b red]🛑 SOFT E-STOP engaged[/b red]"
                if panic:
                    self.ctl.cancel_active_move()
                    msg = "[b red]🛑🛑 PANIC — soft e-stop + cancelled active move[/b red]"
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
                except Exception:  # noqa: BLE001 - polling is best-effort
                    pass
                self._stop.wait(2.0)

        def on_unmount(self) -> None:
            self._stop.set()

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
            ctx = f"[yellow]ctx {pct}%[/yellow]" if pct >= threshold else f"ctx {pct}%"
            title = self.ctl.chat_title() or "new chat"
            self.call_from_thread(
                self.status_widget.update,
                f"{mode} {addr} · {_format_status(state)} · {ctx}",
            )
            self.call_from_thread(self._set_subtitle, title)

        def _set_subtitle(self, title: str) -> None:
            self.sub_title = title

        def _animate_thinking(self) -> None:
            if self._busy:
                frame = SPINNER[self._spin % len(SPINNER)]
                self._spin += 1
                elapsed = time.monotonic() - self._think_start
                self.thinking_widget.update(f"{frame} Thinking… ({elapsed:.0f}s)  [dim]· Esc to e-stop[/dim]")
                self._thinking_shown = True
            elif self._thinking_shown:
                self.thinking_widget.update("")
                self._thinking_shown = False

        # ----- rendering helpers ------------------------------------------ #
        def _write(self, renderable: Any) -> None:
            self.log_widget.write(renderable)

        def _set_idle(self) -> None:
            self._busy = False
            self.thinking_widget.update("")
            self._thinking_shown = False

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
            for call in result.get("tool_calls", []):
                name = call.get("name")
                args = call.get("arguments")
                arg_str = _format_args(args)
                self._write(f"  [magenta]🔧 {name}[/magenta]([cyan]{arg_str}[/cyan])")
                payload = call.get("payload") or {}
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
            for mission_id in result.get("created_mission_ids", []):
                self._write(f"  [blue]＋ mission {mission_id[:8]}[/blue]")
            self._write(f"[b green]agent[/b green] {result.get('assistant_response', '')}")
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
