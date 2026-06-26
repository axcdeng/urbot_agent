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

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.main import ServiceContainer, build_services


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
    "  /status            current robot state",
    "  /markers           known markers and aliases",
    "  /release           release the soft e-stop",
    "  /clear             clear the transcript",
    "  /help              this help",
    "  /quit              exit",
    "",
    "[b]Keys[/b]",
    "  Esc                soft e-stop (immediate)",
    "  Esc Esc            PANIC: soft e-stop + cancel active move",
    "",
    "Anything not starting with '/' is sent to the AI agent.",
]


class AgentConsole:
    """Thin, synchronous wrapper over the service container.

    Every method here is safe to call from a worker thread; none of them
    touch UI state.
    """

    def __init__(self, services: ServiceContainer):
        self.services = services

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

    def chat(self, message: str) -> dict[str, Any]:
        return self.services.agent_planner.run_chat(message)

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
            if not args:
                return CommandResult([f"robot address is {self.robot_addr} (target {self.live_target})"])
            host, _, port = args[0].partition(":")
            host = host.strip()
            if not host:
                return CommandResult(["usage: /ip <host>[:port]"])
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

        if cmd in ("help", "?"):
            return CommandResult(list(HELP_LINES))

        if cmd == "clear":
            return CommandResult(clear=True)

        if cmd in ("quit", "exit", "q"):
            return CommandResult(["bye"], quit=True)

        return CommandResult([f"unknown command: /{cmd} (try /help)"])


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
        #status { height: 1; background: $panel; color: $text; padding: 0 1; }
        #prompt { dock: bottom; }
        """

        BINDINGS = [
            Binding("escape", "estop", "E-STOP", priority=True, show=True),
            Binding("ctrl+c", "quit", "Quit", priority=True, show=True),
        ]

        DOUBLE_ESC_WINDOW = 0.45  # seconds

        def __init__(self, console: AgentConsole):
            super().__init__()
            self.ctl = console
            self._last_esc = 0.0
            self._busy = False
            self._stop = threading.Event()

        # ----- layout ----------------------------------------------------- #
        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Vertical():
                yield RichLog(id="log", markup=True, wrap=True, highlight=False)
                yield Static("", id="status")
                yield Input(placeholder="Message the agent, or /help …", id="prompt")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "alex_agent"
            self.log_widget = self.query_one("#log", RichLog)
            self.status_widget = self.query_one("#status", Static)
            self._write("[dim]Connected. Type a message or /help. Esc = soft e-stop.[/dim]")
            mode = "ON (simulated)" if self.ctl.dry_run else f"OFF — LIVE -> {self.ctl.live_target}"
            self._write(f"[dim]dry-run is {mode}[/dim]")
            self.query_one("#prompt", Input).focus()
            self._background_loop()
            self.set_interval(2.0, self._refresh)

        # ----- input ------------------------------------------------------ #
        def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            event.input.clear()
            if not text:
                return
            if text.startswith("/"):
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
            self._chat_worker(text)

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
            busy = " · [yellow]working…[/yellow]" if self._busy else ""
            self.call_from_thread(
                self.status_widget.update,
                f"{mode} {self.ctl.robot_addr} · {_format_status(state)}{busy}",
            )

        # ----- rendering helpers ------------------------------------------ #
        def _write(self, renderable: Any) -> None:
            self.log_widget.write(renderable)

        def _set_idle(self) -> None:
            self._busy = False

        def _render_result(self, result: dict[str, Any]) -> None:
            for call in result.get("tool_calls", []):
                name = call.get("name")
                args = call.get("arguments")
                self._write(f"  [magenta]⚙ {name}[/magenta] {args}")
                payload = call.get("payload") or {}
                summary = payload.get("status") or payload.get("error_message")
                if summary:
                    self._write(f"    [dim]-> {summary}[/dim]")
            for mission_id in result.get("created_mission_ids", []):
                self._write(f"  [blue]＋ mission {mission_id}[/blue]")
            for task_id in result.get("created_task_ids", []):
                self._write(f"  [blue]＋ task {task_id}[/blue]")
            self._write(f"[b green]agent[/b green] {result.get('assistant_response', '')}")


def main() -> None:
    if not _TEXTUAL_AVAILABLE:
        raise SystemExit(
            "The TUI requires the 'textual' package. Install it with: pip install -e ."
        )
    services = build_services(get_settings())
    AlexAgentTUI(AgentConsole(services)).run()


if __name__ == "__main__":
    main()
