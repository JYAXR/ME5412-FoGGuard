"""Seeded receiver properties, independent of learned weights and patient labels."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fogguard.cueing.replay import replay_guidance
from fogguard.evaluation.automation2 import audit_commands
from fogguard.perception import KEYS, fuse_perception


def random_sources(seed, observations=28):
    """Constructed source predictions; their rule tests are NOT model accuracy."""
    rng = np.random.default_rng(seed)
    fog_parts, direction_parts, clocks = [], [], []
    for record in range(2):
        ends = 2 + np.arange(observations) * 0.5
        common = pd.DataFrame({
            "subject_id": f"fixture-{record}", "session_id": "1", "task_id": "1",
            "window_start_s": ends - 119 / 60, "window_end_s": ends,
            "model_version": "randomized-receiver-fixture-v1",
            "prediction_source": "constructed_random_receiver_fixture",
        })
        scores = rng.choice([0.1, 0.46, 0.8, 0.9], observations)
        # A fixed delay per record preserves source ordering while testing expiry gaps.
        fog = common.assign(available_at_s=ends + float(rng.choice([0, 0.1, 0.5, 0.7])),
                            fog_probability=scores, fog_threshold=0.5, is_fog=scores >= 0.5,
                            input_valid=rng.random(observations) > 0.12)
        direction = common.assign(
            available_at_s=ends + float(rng.choice([0, 0.1, 0.5])),
            direction=rng.choice(["straight", "left", "right", "unknown"], observations),
            direction_confidence=rng.choice([0.4, 0.7, 0.95], observations),
            direction_task="current", horizon_s=0.0,
            input_valid=rng.random(observations) > 0.12,
        )
        # Make one run of valid immediate positive observations in every trial.
        if record == 0:
            fog["available_at_s"] = ends
            direction["available_at_s"] = ends
            fog.loc[:4, ["fog_probability", "is_fog", "input_valid"]] = [0.9, True, True]
            direction.loc[:4, ["direction", "direction_confidence", "input_valid"]] = [
                "straight", 0.95, True]
        fog_parts.append(fog)
        direction_parts.append(direction)
        clocks.append(pd.DataFrame({"subject_id": f"fixture-{record}", "session_id": "1",
                                    "task_id": "1",
                                    "timestamp_s": np.round(np.arange(0, ends[-1] + 1, 0.1), 9)}))
    return pd.concat(fog_parts, ignore_index=True), pd.concat(direction_parts, ignore_index=True), pd.concat(clocks, ignore_index=True)


def _same_record(frame, row):
    result = frame
    for key in KEYS:
        result = result.loc[result[key].astype(str).eq(str(row[key]))]
    return result


def decisions_equal(first, second):
    """CSV-equivalent numbers (0 and 0.0) must not become false failures."""
    try:
        pd.testing.assert_frame_equal(first, second, check_dtype=False, check_exact=True)
    except AssertionError:
        return False
    return True


def receiver_issues(fog, direction, result):
    """Reconstruct latest causal source independently, without event private methods."""
    issues = []
    for index, row in result.iterrows():
        now = float(row.available_at_s)
        fresh = {}
        for name, source in (("fog", fog), ("direction", direction)):
            records = _same_record(source, row)
            arrived = records.loc[records.available_at_s.le(now + 1e-8)]
            latest = None if arrived.empty else arrived.iloc[-1]
            age = round(now - latest.window_end_s, 9) if latest is not None else np.nan
            expected = ("missing" if latest is None else "invalid" if not latest.input_valid
                        else "fresh" if 0 <= age < 0.5
                        else "stale")
            fresh[name] = expected == "fresh"
            if row[f"{name}_status"] != expected:
                issues.append({"row": int(index), "code": f"{name}_source_status"})
            if bool(row[f"{name}_input_fresh"]) != fresh[name]:
                issues.append({"row": int(index), "code": f"{name}_freshness"})
            if row[f"{name}_observation_is_new"] and (
                not fresh[name] or abs(latest.available_at_s - now) > 1e-8
            ):
                issues.append({"row": int(index), "code": f"{name}_held_counted_as_new"})
        if bool(row.input_fresh) != (fresh["fog"] and fresh["direction"]):
            issues.append({"row": int(index), "code": "joint_freshness"})
        if not fresh["fog"] and (
            row.fog_state != "unknown" or row.event_phase != "safe_none"
            or pd.notna(row.is_fog) or pd.notna(row.fog_probability)
        ):
            issues.append({"row": int(index), "code": "unavailable_fog_became_observation"})
        if not fresh["direction"] and row.turn != "unknown":
            issues.append({"row": int(index), "code": "unavailable_direction_retained"})
    for _, group in result.groupby(KEYS, sort=False):
        if not (group.available_at_s.diff().dropna() > 0).all():
            issues.append({"row": None, "code": "nonchronological_output"})
        for position in range(1, len(group)):
            before, after = group.iloc[position-1], group.iloc[position]
            if (after.fog_input_fresh and not after.fog_observation_is_new
                    and before.event_phase != after.event_phase):
                issues.append({"row": int(after.name), "code": "held_row_changed_event_phase"})
    return issues


def run_receiver_trials(output, *, seed=20261030, trials=32, observations=28):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for trial in range(trials):
        fog, direction, clock = random_sources(seed + trial, observations)
        result = fuse_perception(fog, direction, clock=clock)
        issues = receiver_issues(fog, direction, result)
        result.to_csv(output / f"trial_{trial:03d}_perception.csv", index=False)
        fog.to_csv(output / f"trial_{trial:03d}_fog.csv", index=False)
        direction.to_csv(output / f"trial_{trial:03d}_direction.csv", index=False)
        clock.to_csv(output / f"trial_{trial:03d}_clock.csv", index=False)
        rows.append({"trial": trial, "check": "causality_freshness_new_observation",
                     "status": "FAIL" if issues else "PASS", "rows": len(result),
                     "details": json.dumps(issues)})
        # Spoofed annotation columns must not change any runtime output.
        poisoned = fuse_perception(fog.assign(truth_fog_label=1, fog=1),
                                   direction.assign(truth_direction="right", activity=7), clock=clock)
        invariant = poisoned.equals(result)
        rows.append({"trial": trial, "check": "truth_injection_invariance",
                     "status": "PASS" if invariant else "FAIL", "rows": len(result), "details": ""})
        # Modifying not-yet-arrived scores/directions cannot alter a prefix.
        cutoff = 5.0
        altered_fog, altered_direction = fog.copy(), direction.copy()
        altered_fog.loc[altered_fog.available_at_s.gt(cutoff), ["fog_probability", "is_fog"]] = [0.1, False]
        altered_direction.loc[altered_direction.available_at_s.gt(cutoff), "direction"] = "right"
        altered = fuse_perception(altered_fog, altered_direction, clock=clock)
        # IDs are enumerated globally across complete record batches; compare decisions,
        # not cosmetic event numbers allocated after processing another record.
        prefix_columns = [c for c in result if c != "event_id"]
        invariant = result.loc[result.available_at_s.le(cutoff), prefix_columns].equals(
            altered.loc[altered.available_at_s.le(cutoff), prefix_columns])
        rows.append({"trial": trial, "check": "future_change_prefix_invariance",
                     "status": "PASS" if invariant else "FAIL", "rows": len(result),
                     "details": "Compare causal decisions; exclude globally enumerated event IDs"})
        # Other-record state cannot change a record's phases/turns (IDs are globally numbered).
        alone = fuse_perception(fog.loc[fog.subject_id.eq("fixture-1")],
                                direction.loc[direction.subject_id.eq("fixture-1")],
                                clock=clock.loc[clock.subject_id.eq("fixture-1")])
        columns = ["available_at_s", "event_phase", "stable_fog_detected", "turn", "turn_confidence"]
        invariant = decisions_equal(
            result.loc[result.subject_id.eq("fixture-1"), columns].reset_index(drop=True),
            alone[columns].reset_index(drop=True))
        rows.append({"trial": trial, "check": "record_isolation",
                     "status": "PASS" if invariant else "FAIL", "rows": len(alone),
                     "details": "Exact decision values; allow numeric dtype representation differences"})
        commands = replay_guidance(result)
        lifecycle = audit_commands(commands)
        invalid_active = commands.guidance_visual_enabled & (
            ~commands.input_fresh | commands.turn.eq("unknown"))
        rows.append({"trial": trial, "check": "downstream_lifecycle_and_unknown_withdrawal",
                     "status": "FAIL" if lifecycle or invalid_active.any() else "PASS",
                     "rows": len(commands), "details": json.dumps(lifecycle)})
        commands.to_csv(output / f"trial_{trial:03d}_commands.csv", index=False)
        invalid_sources = {
            "reverse_source_time": fog.iloc[::-1].copy(),
            "duplicate_source_time": pd.concat([fog.iloc[:1], fog], ignore_index=True),
            "missing_record_key": fog.drop(columns="subject_id"),
            "availability_before_input": fog.assign(available_at_s=fog.window_end_s - 0.1),
            "nonfinite_score": fog.assign(fog_probability=np.inf, input_valid=True),
        }
        for label, invalid in invalid_sources.items():
            try:
                fuse_perception(invalid, direction)
                status, details = "FAIL", "Invalid input accepted"
            except (ValueError, TypeError) as error:
                status, details = "PASS", str(error)
            rows.append({"trial": trial, "check": label, "status": status, "rows": 0,
                         "details": details})
    table = pd.DataFrame(rows)
    table.to_csv(output / "checks.csv", index=False)
    summary = {"scope": "constructed source predictions; no trained model accuracy measured",
               "seed": seed, "trials": trials, "records_per_trial": 2,
               "checks": len(table), "pass": int(table.status.eq("PASS").sum()),
               "fail": int(table.status.eq("FAIL").sum()),
               "perception_rows": int(table.loc[table.check.eq(
                   "causality_freshness_new_observation"), "rows"].sum())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
