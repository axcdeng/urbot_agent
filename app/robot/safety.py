from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.water.normalizer import NormalizedRobotState


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str
    code: str


class SafetyValidator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_move(
        self,
        state: NormalizedRobotState,
        *,
        target_exists: bool = True,
        allow_interruption: bool = False,
    ) -> SafetyDecision:
        if not state.online:
            return SafetyDecision(False, "Robot is offline.", "offline")
        if state.estop_state:
            return SafetyDecision(False, "Robot is in emergency stop.", "estop")
        if state.battery_percent is not None and state.battery_percent < self.settings.min_move_battery_percent:
            return SafetyDecision(False, "Battery is below the minimum threshold.", "low_battery")
        if state.error_code and state.error_code != "00000000":
            return SafetyDecision(False, "Robot reports an error state.", "robot_error")
        if state.move_status == "running" and not allow_interruption:
            return SafetyDecision(False, "Robot is already moving.", "already_moving")
        if not target_exists:
            return SafetyDecision(False, "Requested target does not exist.", "unknown_target")
        return SafetyDecision(True, "Move is allowed.", "ok")

    def validate_release_estop(self, confirmed: bool) -> SafetyDecision:
        if not confirmed:
            return SafetyDecision(False, "Explicit confirmation is required to release emergency stop.", "confirmation_required")
        return SafetyDecision(True, "Emergency stop release confirmed.", "ok")

    def allow_emergency_stop(self) -> SafetyDecision:
        return SafetyDecision(True, "Emergency stop is always allowed.", "ok")
