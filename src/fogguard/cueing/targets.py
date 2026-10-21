"""Local illustrative foot targets, generated only at start or accepted landing."""

import math

from fogguard.cueing.contracts import FootTarget, GuidanceConfig


def make_targets(
    config: GuidanceConfig,
    *,
    command_id: str,
    first_index: int,
    first_foot: str,
    direction: str,
    step_length_m: float,
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[FootTarget, ...]:
    """x forward / y left / yaw positive left; origin is configured, not measured."""
    x, y, yaw = anchor
    sign = {"straight": 0, "left": 1, "right": -1}[direction]
    targets = []
    for offset in range(config.target_count):
        foot = first_foot if offset % 2 == 0 else (
            "right" if first_foot == "left" else "left"
        )
        rotation = sign * step_length_m / config.turn_radius_m
        new_yaw = min(math.pi / 2, max(-math.pi / 2, yaw + rotation))
        middle_yaw = (yaw + new_yaw) / 2
        x += step_length_m * math.cos(middle_yaw)
        y += step_length_m * math.sin(middle_yaw)
        yaw = new_yaw
        lateral = (1 if foot == "left" else -1) * config.step_width_m / 2
        targets.append(FootTarget(
            target_id=f"{command_id}/step-{first_index + offset}", foot=foot,
            x_m=x - lateral * math.sin(yaw),
            y_m=y + lateral * math.cos(yaw), yaw_deg=math.degrees(yaw),
        ))
    return tuple(targets)
