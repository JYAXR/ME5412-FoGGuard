"""Optional simulated marching extension. Original controller remains the baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from fogguard.cueing.contracts import FootTarget, GuidanceConfig
from fogguard.cueing.guidance import GuidanceController
from fogguard.cueing.targets import make_targets


@dataclass(frozen=True)
class MarchConfig(GuidanceConfig):
    enable_march_fallback: bool = True
    march_required_steps: int = 4
    march_maximum_forward_m: float = 0.08

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.march_required_steps, int) or self.march_required_steps < 2:
            raise ValueError("march_required_steps must be an integer >= 2")
        if not 0 < self.march_maximum_forward_m < self.minimum_forward_m:
            raise ValueError("March displacement must be positive and below a forward step")


class MarchGuidanceController(GuidanceController):
    """Reuse original safety, lifecycle, fading and spacing; add stationary responses.

    Four reliable alternating stationary responses return to forward targets.
    Marching never supplies forward recovery evidence or advances the anchor.
    All positions remain illustrative configured coordinates, not measured pose.
    """

    def __init__(self, config=None, *, responses_configured=False):
        super().__init__(config or MarchConfig(), responses_configured=responses_configured)

    @property
    def march_enabled(self):
        return isinstance(self.config, MarchConfig) and self.config.enable_march_fallback

    def _clear_episode(self):
        super()._clear_episode()
        self._marching = False
        self._march_count = 0
        self._march_last_time = None
        self._march_serial = 0
        self._local_anchor = (0.0, 0.0, 0.0)

    def _enter_march(self):
        if self._marching:
            return
        self._marching = True
        self._march_count = 0
        self._march_last_time = None
        self._march_foot = self._targets[0].foot if self._targets else self.config.first_foot
        self._stationary_targets()

    def _stationary_targets(self):
        x, y, yaw = self._local_anchor
        feet = (self._march_foot, "right" if self._march_foot == "left" else "left")
        self._targets = tuple(
            FootTarget(
                f"{self._command_id}/march-{self._march_serial}-{foot}", foot,
                x - (1 if foot == "left" else -1) * self.config.step_width_m / 2 * math.sin(yaw),
                y + (1 if foot == "left" else -1) * self.config.step_width_m / 2 * math.cos(yaw),
                math.degrees(yaw),
            ) for foot in feet
        )

    def _leave_march(self, item):
        self._marching = False
        self._clear_recovery_evidence()
        self._state = "GUIDING"
        self._targets = make_targets(
            self.config, command_id=self._command_id, first_index=self._target_index,
            first_foot=self._march_foot, direction=item.turn,
            step_length_m=self._step_length, anchor=self._local_anchor,
        )

    def _accept_responses(self, item, responses):
        if not self.march_enabled:
            return super()._accept_responses(item, responses)
        self._current_item = item
        reason = ""
        for response in responses:
            if self._state == "RESET_CUE":
                self._enter_march()
            reliable = (
                response.reliability >= self.config.minimum_response_reliability
                and response.response_status in {"success", "delayed"}
                and self._started is not None and response.timestamp_s >= self._started
                and item.now_s - response.timestamp_s <= self.config.maximum_step_gap_s
            )
            if self._marching and reliable:
                if (response.foot != self._march_foot or response.forward_m is None
                        or abs(response.forward_m) > self.config.march_maximum_forward_m):
                    self._march_count = 0
                    self._march_last_time = None
                    reason = "march_response_rejected"
                    continue
                if (self._march_last_time is not None and response.timestamp_s
                        - self._march_last_time > self.config.maximum_step_gap_s):
                    self._march_count = 0
                self._march_count += 1
                self._march_last_time = response.timestamp_s
                self._march_foot = "right" if response.foot == "left" else "left"
                self._march_serial += 1
                self._stationary_targets()
                reason = "simulated_stationary_step_accepted"
                if self._march_count >= self.config.march_required_steps:
                    self._leave_march(item)
                    reason = "march_completed_return_forward"
                continue
            landed = self._targets[0] if self._targets else None
            reason = super()._accept_responses(item, (response,)) or reason
            if self._marching and response.response_status in {"unable", "no_response"}:
                self._march_count = 0
                self._march_last_time = None
            if self._marching and (not self._response_usable or self._state != "RESET_CUE"):
                self._leave_march(item)
            if reason == "simulated_forward_step_accepted" and landed is not None:
                yaw = math.radians(landed.yaw_deg)
                lateral = (1 if landed.foot == "left" else -1) * self.config.step_width_m / 2
                self._local_anchor = (landed.x_m + lateral * math.sin(yaw),
                                      landed.y_m - lateral * math.cos(yaw), yaw)
            if self._state == "RESET_CUE":
                self._enter_march()
                reason = reason.replace("countdown", "march")
        return reason

    def _stop(self, now, reason, state="IDLE"):
        result = super()._stop(now, reason, state)
        return replace(result, schema_version="guidance-march-v1") if self.march_enabled else result

    def _emit(self, now, reason, starting):
        if not self.march_enabled:
            return super()._emit(now, reason, starting)
        if self._state == "RESET_CUE":
            self._enter_march()
            reason = reason.replace("countdown", "march")
        elif self._marching:
            self._leave_march(self._current_item)
            reason = reason.replace("countdown_completed", "march_timeout")
        previous = self._previous_output
        result = super()._emit(now, reason, starting)
        if self._marching:
            result = replace(result, control_state="MARCH_FALLBACK", audio_mode="march_alternating")
        result = replace(result, schema_version="guidance-march-v1")
        if not starting:
            changed = previous is None or any(
                getattr(result, name) != getattr(previous, name) for name in (
                    "control_state", "targets", "visual_opacity", "audio_volume", "audio_mode",
                )
            )
            result = replace(result, action="update" if changed else "hold")
        self._previous_output = result
        return result
