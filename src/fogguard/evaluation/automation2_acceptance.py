"""Acceptance gates over saved outputs, independent of controller decisions.

Known recognition limitations are reported explicitly and never called passes.
Missing required evidence is a gate failure, not an implicit pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def check_acceptance(directory: Path) -> list[dict]:
    results = []

    def check(name, case, predicate, meaning, strategy="response"):
        try:
            folder = directory / "timelines" / case
            frame = pd.read_csv(folder / f"{strategy}.csv")
            if frame.empty:
                raise ValueError("No command rows; empty evidence cannot pass")
            config = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))[
                strategy]["config"]
            valid = bool(predicate(frame, config))
            results.append({"check": name, "case": case, "strategy": strategy,
                            "status": "PASS" if valid else "FAIL", "meaning": meaning,
                            "evidence": f"timelines/{case}/{strategy}.csv"})
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            results.append({"check": name, "case": case, "strategy": strategy,
                            "status": "FAIL", "meaning": meaning,
                            "error": f"{type(exc).__name__}: {exc}"})

    def no_start(frame, _):
        return not frame.guidance_action.eq("start").any()

    for name in ("normal", "normal_right_turn", "single_candidate"):
        check(f"negative_{name}", f"constructed_{name}", no_start, "负例不启动救援提示")
    for name in ("normal", "normal_right_turn"):
        check(f"sensor_negative_{name}", f"sensor_{name}", no_start, "传感器负例不误提示")

    def single_episode(frame, _):
        return frame.guidance_action.eq("start").sum() == 1 and (
            frame.guidance_action.eq("stop").sum() == 1)

    for name in ("straight_fog", "turning_fog", "right_turning_fog", "fog_recovery"):
        check(f"sensor_episode_{name}", f"sensor_{name}", single_episode,
              "核心传感器冻结场景形成一次提示并明确停止")
    check("short_dip", "constructed_short_dip", single_episode,
          "短暂评分回落维持同一提示，不重复启动")

    def target_turn(sign):
        def correct(frame, _):
            starts = frame.loc[frame.guidance_action.eq("start")]
            if starts.empty:
                return False
            targets = json.loads(starts.iloc[0].guidance_targets)
            return bool(targets) and all(sign * item["yaw_deg"] > 0 for item in targets)
        return correct

    for name, sign in (("left_fog", 1), ("right_fog", -1)):
        check(f"direction_{name}", f"constructed_{name}", target_turn(sign),
              "转弯脚印朝向符号正确")

    def safe(frame, _):
        unsafe = frame.loc[frame.guidance_control_state.eq("SAFE_NONE")]
        stops = unsafe.loc[unsafe.guidance_action.eq("stop")]
        return len(stops) > 0 and not unsafe.guidance_visual_enabled.any() and (
            not unsafe.guidance_audio_enabled.any()) and unsafe.guidance_targets.eq("[]").all()

    for name in ("unknown_direction", "stale_input"):
        check(f"safety_{name}", f"constructed_{name}", safe,
              "活动提示遇到不可靠输入后撤销双模态及目标")

    def faded(frame, _):
        fade = frame.loc[frame.guidance_control_state.eq("FADING")]
        return (len(fade) > 0 and fade.guidance_visual_opacity.between(0, 1, inclusive="neither").any()
                and frame.guidance_reason.eq("simulated_recovery_fade_completed").any())

    check("fading", "constructed_successful_recovery", faded, "可靠模拟落脚后逐渐减弱并停止")
    check("no_fade_factor", "constructed_successful_recovery",
          lambda frame, _: not frame.guidance_control_state.eq("FADING").any(),
          "只关闭渐弱后不进入渐弱状态", strategy="no_fade")

    def restored(frame, config):
        rows = frame.loc[frame.guidance_reason.eq("fog_reconfirmed_restore_intensity")]
        return len(rows) > 0 and rows.guidance_visual_opacity.eq(config["opacity"]).all() and (
            rows.guidance_audio_volume.eq(config["volume"]).all())

    check("refreeze", "constructed_refreeze_during_fade", restored,
          "渐弱期间再次冻结恢复配置强度")

    def timeout(frame, config):
        starts = frame.loc[frame.guidance_action.eq("start")]
        stops = frame.loc[frame.guidance_reason.eq("episode_timeout")]
        countdown = frame.guidance_control_state.eq("RESET_CUE")
        entries = (countdown & ~countdown.shift(fill_value=False)).sum()
        if len(starts) != 1 or len(stops) != 1 or entries != 1:
            return False
        elapsed = stops.iloc[0].guidance_timestamp_s - starts.iloc[0].guidance_timestamp_s
        return elapsed == config["maximum_episode_s"]

    check("bounded_no_response", "constructed_persistent_no_response", timeout,
          "一次倒计时备用，达到时长上限停止，同事件不重启")
    check("missing_response", "constructed_response_missing",
          lambda frame, _: "RESET_CUE" not in set(frame.guidance_control_state)
          and frame.guidance_response_source.eq("unavailable_fixed_fallback").any(),
          "响应源缺失退回固定提示，不冒充无响应")

    def spacing(frame, config):
        active = frame.loc[frame.guidance_visual_enabled]
        lengths = active.guidance_step_length_m
        deltas = active.groupby("guidance_command_id", sort=False).guidance_step_length_m.diff()
        return not lengths.empty and lengths.max() > config["step_length_m"] and (
            lengths.between(config["minimum_step_length_m"], config["maximum_step_length_m"]).all()
            and deltas.dropna().abs().le(config["maximum_step_adjustment_m"] + 1e-9).all())

    check("spacing_bounds", "constructed_adaptive_spacing", spacing,
          "可选间距确实变化，且满足范围和单次变化限制", strategy="spacing")
    check("stationary_suppression_reference", "sensor_voluntary_stop_suppression_on", no_start,
          "可选静止抑制对照不启动提示；不改变默认配置")
    try:
        frame = pd.read_csv(directory / "timelines/sensor_voluntary_stop/response.csv")
        count = int(frame.guidance_action.eq("start").sum())
        results.append({"check": "stationary_default_limit", "case": "sensor_voluntary_stop",
                        "strategy": "response", "status": "KNOWN_LIMITATION" if count else "PASS",
                        "meaning": f"默认策略主动停步误启动 {count} 次；保留原规则并披露",
                        "evidence": "timelines/sensor_voluntary_stop/response.csv"})
    except (OSError, ValueError, KeyError) as exc:
        results.append({"check": "stationary_default_limit", "status": "FAIL", "error": str(exc)})
    return results


def acceptance_summary(results: list[dict]) -> dict:
    counts = {status: sum(item["status"] == status for item in results)
              for status in ("PASS", "FAIL", "KNOWN_LIMITATION")}
    return {"counts": counts, "regression_gate_passed": bool(results) and counts["FAIL"] == 0,
            "scope": "工程回归门槛；通过不表示已知识别局限解决，也不代表全组最终验收"}
