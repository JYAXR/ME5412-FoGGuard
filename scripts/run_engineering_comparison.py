"""Paired deterministic engineering comparison; no patient response/efficacy model."""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.complete import strategy_configs
from fogguard.context import FogStarTemporalIntegrator
from fogguard.cueing.march import MarchGuidanceController
from fogguard.cueing.replay import replay_guidance, summarize_guidance
from fogguard.cueing.scenarios import _predictions, _steps
from fogguard.evaluation.automation2 import audit_commands, evaluate_timeline

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"fixed": "固定提示", "countdown": "倒计时备用", "march": "踏步完整方案",
         "no_fade": "关闭渐弱", "no_spacing": "固定间距"}


def make_cases():
    """Grid points are designed probes, not random samples or independent patients."""
    specs = []
    for direction, onset, duration, interval, length in itertools.product(
        ("straight", "left", "right"), (4, 8), (12, 20), (0.5, 1.0), (0.24, 0.40, 0.52)
    ):
        specs.append({"family": "forward", "direction": direction, "onset": onset,
                          "duration": duration, "interval": interval, "length": length})
    for onset, duration, interval in itertools.product((4, 8), (12, 20), (0.5, 1.0)):
        specs.append({"family": "march", "onset": onset, "duration": duration, "interval": interval})
    for family, onset, duration in itertools.product(
        ("stale_input", "unknown_direction", "missing_response", "wrong_foot",
         "voluntary_false_positive", "persistent_no_response", "refreeze"),
        (4, 8), (12, 20)
    ):
        specs.append({"family": family, "onset": onset, "duration": duration})
    for family, onset in itertools.product(("normal", "single_candidate"), (4, 8)):
        specs.append({"family": family, "onset": onset, "duration": 1})
    for index, spec in enumerate(specs):
        spec = {"direction": "straight", "interval": 0.5, "length": 0.4} | spec
        name = f"{index:03d}_{spec['family']}"
        family, onset, duration = spec["family"], spec["onset"], spec["duration"]
        if family == "persistent_no_response":
            duration += 24  # 18/22 s positive support exceeds the 15 s episode cap.
            spec["duration"] = duration
        positive = range(onset, onset + duration) if family != "normal" else range(0)
        raw = _predictions(name, direction=spec["direction"], positive=positive, count=88)
        start = 2 + onset * 0.5
        responses = None
        response_start = start + 1
        if family in {"forward", "refreeze", "wrong_foot"}:
            times = [response_start + spec["interval"] * i for i in range(28)]
            responses = _steps(name, times, lengths=[spec["length"]] * len(times))
            if family == "wrong_foot":
                responses["foot"] = "right"
        if family in {"march", "persistent_no_response", "missing_response"}:
            status = "missing_input" if family == "missing_response" else "no_response"
            first = {"scenario_id": name, "timestamp_s": response_start, "foot": "none",
                         "forward_m": None, "lateral_m": None, "response_status": status,
                         "delay_s": None, "reliability": 1.0, "source": "simulated"}
            rows = [first]
            if family == "march":
                times = [response_start + spec["interval"] * (i + 1) for i in range(4)]
                rows += _steps(name, times, lengths=[0.0] * 4).to_dict("records")
                times = [times[-1] + spec["interval"] * (i + 1) for i in range(24)]
                rows += _steps(name, times).to_dict("records")
            responses = pd.DataFrame(rows)
        if family == "voluntary_false_positive":
            raw["truth_fog_label"] = 0
            raw["motion_state"], raw["motion_confidence"] = "stationary", 0.9
        if family == "refreeze":
            raw.loc[onset + 5:onset + 7, ["fog_probability", "is_fog",
                                        "truth_fog_label"]] = [0.1, False, 0]
        if family == "unknown_direction":
            raw.loc[onset + 4:onset + 6, "direction"] = "other"
        if family == "stale_input":
            raw["received_at_s"] = raw.window_end_s
            raw.loc[onset + 4, "received_at_s"] += 2
        events = FogStarTemporalIntegrator().process(raw)
        yield name, spec, raw, events, responses


def measure(events, commands, spec):
    metrics = evaluate_timeline(events, commands)
    summary = summarize_guidance(commands)
    windows = commands.loc[~commands.guidance_terminal].copy()
    t = windows.window_end_s.to_numpy(float)
    dt = np.minimum(np.maximum(np.diff(t, append=t[-1]), 0), 0.5)
    active = windows.guidance_visual_enabled.to_numpy(bool)
    truth = windows.truth_fog_label.to_numpy(int)
    # Input-window truth held forward for <=0.5 s; no onset backdating.
    after_end = t >= (float(t[truth == 1][-1]) + 0.5 if (truth == 1).any() else np.inf)
    invalid = (~windows.input_fresh.astype(bool) | ~windows.turn.isin(
        ["straight", "left", "right"]) | windows.input_age_s.gt(1)).to_numpy(bool)
    starts = windows.loc[windows.guidance_action.eq("start")]
    stops = commands.loc[commands.guidance_action.eq("stop")]
    lengths = windows.guidance_step_length_m.to_numpy(float)
    accepted = windows.guidance_reason.eq("simulated_forward_step_accepted").to_numpy()
    spacing_error = None
    if spec["family"] == "forward" and accepted.any():
        desired = np.clip(spec["length"], 0.32, 0.55)
        spacing_error = float(np.mean(np.abs(lengths[accepted] - desired)))
    # First targets should follow configured turn sign; geometry is illustrative.
    turn_checks = turn_errors = 0
    for row in windows.loc[windows.guidance_visual_enabled].itertuples():
        targets = json.loads(row.guidance_targets)
        if row.turn in {"left", "right"} and targets:
            turn_checks += 1
            sign = 1 if row.turn == "left" else -1
            turn_errors += int(targets[0]["yaw_deg"] * sign <= 0)
    return {
        "truth_events": metrics["truth_events"],
        "missed_truth_events": metrics["unprompted_truth_events"],
        "unmatched_starts": metrics["unmatched_guidance_starts"],
        "starts_on_non_fog_windows": metrics["starts_on_non_fog_windows"],
        "truth_to_start_s": metrics["mean_truth_window_to_guidance_delay_s"],
        "confirmed_to_start_s": summary["mean_confirmed_to_start_delay_s"],
        "active_s": float(np.sum(dt * active)),
        "active_non_fog_s": float(np.sum(dt * active * (truth == 0))),
        "after_final_truth_end_active_s": float(np.sum(dt * active * after_end)),
        "invalid_input_active_rows": int(np.sum(active * invalid)),
        "start_count": len(starts), "stop_count": len(stops),
        "timeout_stops": int(stops.guidance_reason.eq("episode_timeout").sum()),
        "censored_stops": int(stops.guidance_reason.eq("replay_ended").sum()),
        "fade_stops": int(stops.guidance_reason.eq("simulated_recovery_fade_completed").sum()),
        "march_completions": int(windows.guidance_reason.eq(
            "march_completed_return_forward").sum()),
        "march_mode_timeouts": int(windows.guidance_reason.eq(
            "march_timeout_fixed_guidance").sum()),
        "unclosed_commands": summary["unclosed_commands"],
        "spacing_error_m": spacing_error,
        "minimum_spacing_m": float(lengths.min()),
        "maximum_spacing_m": float(lengths.max()),
        "spacing_out_of_bounds_rows": int(((lengths < 0.32-1e-9)
                                            | (lengths > 0.55+1e-9)).sum()),
        "turn_checks": turn_checks, "turn_errors": turn_errors,
        "stop_reasons": json.dumps(summary["stop_reasons"], sort_keys=True),
    }


def paired_results(frame):
    rows = []
    for family in frame.family.unique():
        subset = frame.loc[frame.family.eq(family)]
        for reference, alternative in (("march", "countdown"), ("march", "no_fade"),
                                       ("march", "no_spacing"), ("march", "fixed")):
            a = subset.loc[subset.strategy.eq(reference)].set_index("case")
            b = subset.loc[subset.strategy.eq(alternative)].set_index("case").reindex(a.index)
            for metric in ("active_s", "after_final_truth_end_active_s", "truth_to_start_s",
                           "spacing_error_m", "unmatched_starts", "missed_truth_events"):
                diff = (a[metric] - b[metric]).dropna()
                if not len(diff):
                    continue
                rows.append({"family": family, "reference": reference, "alternative": alternative,
                                 "metric": metric, "pairs": len(diff),
                                 "reference_mean": float(a.loc[diff.index, metric].mean()),
                                 "alternative_mean": float(b.loc[diff.index, metric].mean()),
                                 "mean_paired_difference": float(diff.mean()),
                                 "negative": int((diff < -1e-9).sum()),
                                 "equal": int((diff.abs() <= 1e-9).sum()),
                                 "positive": int((diff > 1e-9).sum())})
    return pd.DataFrame(rows)


def run(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    inputs, timelines = output / "inputs", output / "timelines"
    inputs.mkdir(exist_ok=True)
    timelines.mkdir(exist_ok=True)
    rows, specs, failures = [], [], []
    strategies = strategy_configs()
    for name, spec, raw, events, responses in make_cases():
        specs.append(dict(case=name, **spec))
        raw.to_csv(inputs / f"{name}_candidates.csv", index=False)
        events.to_csv(inputs / f"{name}_events.csv", index=False)
        if responses is not None:
            responses.to_csv(inputs / f"{name}_responses.csv", index=False)
        for strategy, config in strategies.items():
            commands = replay_guidance(events, config=config, responses=responses,
                                       controller_type=MarchGuidanceController)
            commands.to_csv(timelines / f"{name}_{strategy}.csv", index=False)
            issues = audit_commands(commands)
            failures.extend(dict(case=name, strategy=strategy, **issue) for issue in issues)
            rows.append(dict(case=name, strategy=strategy, **spec,
                             **measure(events, commands, spec), audit_failures=len(issues)))
        if len(specs) % 40 == 0:
            print(f"Completed {len(specs)} cases / {len(rows)} paired strategy runs", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "runs.csv", index=False)
    pd.DataFrame(specs).to_csv(output / "cases.csv", index=False)
    pairs = paired_results(frame)
    pairs.to_csv(output / "paired_comparisons.csv", index=False)
    aggregate = frame.groupby(["family", "strategy"], sort=False).agg(
        runs=("case", "count"), starts=("start_count", "sum"),
        missed_truth_events=("missed_truth_events", "sum"),
        unmatched_starts=("unmatched_starts", "sum"),
        mean_active_s=("active_s", "mean"),
        mean_recovery_tail_s=("after_final_truth_end_active_s", "mean"),
        mean_truth_to_start_s=("truth_to_start_s", "mean"),
        mean_spacing_error_m=("spacing_error_m", "mean"),
        timeout_stops=("timeout_stops", "sum"),
        invalid_active=("invalid_input_active_rows", "sum"),
        audit_failures=("audit_failures", "sum"),
        turn_errors=("turn_errors", "sum"),
    ).reset_index()
    aggregate.to_csv(output / "family_summary.csv", index=False)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    sources = set(tracked) | {str(Path(__file__).relative_to(ROOT)).replace("\\", "/")}
    code_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                   for p in sorted(sources) if (ROOT / p).is_file()}
    files = {str(p.relative_to(output)).replace("\\", "/"):
             hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob("*.csv"))}
    manifest = {"schema": "fogguard-engineering-comparison-v1", "python": platform.python_version(),
                    "cases": len(specs), "runs": len(rows), "seed": None,
                    "design": "Exhaustive deterministic grid, no random sampling or fitted responses",
                    "strategies": {k: v.as_dict() for k, v in strategies.items()},
                    "scope": "Constructed candidate -> actual temporal integrator -> actual controller",
                    "patient_efficacy_available": False, "audit_failures": failures,
                    "source_sha256": code_hashes, "csv_sha256": files,
                    "git_head": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    report = ["# 提示方案工程对照实验", "",
              f"共 {len(specs)} 个参数化案例 × 5 种方案 = {len(rows)} 次运行。",
              "同一案例的候选、真值、落脚响应在各方案间完全相同；响应不由提示生成。",
              "这是工程行为比较，不是患者疗效、冻结识别准确率或硬件实时性能实验。", "",
              "## 怎样比较", "",
              ("72 个向前案例：起始位置 2 种 × 冻结持续 6/10 秒 × 迈步间隔 0.5/1 秒 × "
              "步长 24/40/52 cm × 直行/左转/右转。另有踏步、无响应、缺失响应、"
              "核心输入过期、方向未知、错误脚别、主动停步误报、再冻结、正常和短候选。"),
              "五方案配置固定，不为结果调参。固定方案同时关闭多个功能；其余为单开关对照。",
              "结果按场景分组，不用随意加权的总分宣布冠军；网格不是患者样本，不计算临床置信区间。",
              "",
              "## 核查", "",
              (f"命令审计异常：{int(frame.audit_failures.sum())}；未关闭命令："
              f"{int(frame.unclosed_commands.sum())}；"
              f"无效核心输入仍显示提示的行数：{int(frame.invalid_input_active_rows.sum())}；"
              f"间距越界行数：{int(frame.spacing_out_of_bounds_rows.sum())}；"
              f"回放截断停止：{int(frame.censored_stops.sum())}。"), "",
              "## 先看结论", "",
              ("1. 在持续向前响应的案例中，关闭渐弱可能让提示一直保留至 15 秒超时。"
               "比较下表的恢复尾部时间；这是退出规则差异，不是人体恢复速度。"
               "固定提示的退出同样及时，因此不能说完整方案在时长上全面优于固定提示。"),
              ("2. 开启自适应间距后，输出更接近预设且裁剪后的步长目标。"
               "这是输出跟随性比较，不是病人步态改善。"),
              ("3. 踏步没有显示提示时长优势。每步 1 秒时，四次原地踏步需要 4 秒，"
               "当前备用窗口只有 3 秒，会在完成前退出踏步模式。"
               "这是需要队友评审的参数边界；实验没有修改原参数。"),
              ("4. 主动停步故意误判的案例中，五种方案都产生错误提示。"
               "下游提示策略未能修正这类上游误判，不能隐藏这个失败结果。"),
              ("5. 正常和单次候选不启动；持续无响应由 15 秒上限退出。"
               "输入过期/方向未知先撤销，再确认后重新启动。"
               "未匹配启动也包括重新启动，不能一律计为非冻结误报。"),
              ("6. 不选总冠军：渐弱改善特定持续响应条件下的退出，间距自适应改善设定目标跟随；"
               "踏步增加了可选流程，但暴露慢节奏超时边界。"), "",
              "## 单因素比较（踏步完整方案 vs 关闭对应功能）", "",
              "差值为完整方案减对照；小于零只代表该数值较小，不自动等于效果更好。", "",
              "|场景|对照|指标|配对数|完整方案均值|对照均值|差值|", "|---|---|---|---:|---:|---:|---:|"]
    selected = pairs.loc[
        (pairs.family.eq("forward") & pairs.alternative.eq("no_fade")
         & pairs.metric.eq("after_final_truth_end_active_s"))
        | (pairs.family.eq("forward") & pairs.alternative.eq("no_spacing")
           & pairs.metric.eq("spacing_error_m"))
        | (pairs.family.eq("march") & pairs.alternative.eq("countdown")
           & pairs.metric.eq("active_s"))]
    for row in selected.itertuples():
        report.append(f"|{row.family}|{NAMES[row.alternative]}|{row.metric}|{row.pairs}|"
                      f"{row.reference_mean:.4f}|{row.alternative_mean:.4f}|"
                      f"{row.mean_paired_difference:.4f}|")
    report += ["", "## 全部分组结果", "", ("|场景|方案|次数|未匹配启动|漏提示事件|"
               "平均提示秒|平均恢复尾部秒|超时退出|"), "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.itertuples():
        report.append(f"|{row.family}|{NAMES[row.strategy]}|{row.runs}|{row.unmatched_starts}|"
                      f"{row.missed_truth_events}|{row.mean_active_s:.3f}|"
                      f"{row.mean_recovery_tail_s:.3f}|{row.timeout_stops}|")
    report += ["", "## 指标边界与复查", "",
               "- 真值是构造候选输出时刻的标注，按不超过 0.5 秒保持；延迟不是从原始人体冻结开始计时。",
               "- 恢复尾部：最终真值冻结结束后仍显示提示的累计时间；不是患者恢复用时。",
               ("- 间距误差：已接受向前落脚时，输出间距与裁剪至 32–55 cm 的预设步长的绝对差。"
               "24 cm 输入目标裁剪为 32 cm。固定方案不消费落脚，记为不可用而不是零误差。"),
               "- 非冻结时的活动时间包含正常退出尾部，不全部算误报。未匹配启动按独立真值事件一对一匹配。",
               "- 主动停步案例故意给出错误的冻结候选，用来检查下游是否抑制；不是现实误报率估计。",
               "- 数据缺失分两类：落脚输入缺失允许退化为固定提示；核心状态过期/方向未知应撤销提示。",
               "- 转弯检查只验证首个配置目标朝向的左右符号，不验证真实投影几何或人体安全。",
               ("- 再冻结可能与已有提示重叠；已有提示跨越新事件时，一对一启动匹配可能记漏启动。"
               "需结合时间线，不能直接等同于完全没有提示。"),
               ("- runs.csv 每次运行指标；paired_comparisons.csv 同案例配对差；"
               "inputs/ 原输入；timelines/ 全部命令；manifest.json 配置和哈希。"),
               "", "复现：`python -X utf8 scripts/run_engineering_comparison.py --output-dir <新目录>`"]
    (output / "结果报告.md").write_text("\n".join(report), encoding="utf-8")
    # A static results document, not an interactive demo requiring every case to be clicked.
    body = "<h1>提示方案工程对照结果</h1><p>" + html.escape(report[2]) + "</p>"
    body += "<p>构造输入与预设响应；结果不代表患者疗效。具体定义见同目录结果报告.md。</p>"
    body += "<h2>结论</h2>" + "".join(
        "<p>" + html.escape(line) + "</p>" for line in report
        if line.startswith(tuple(f"{i}. " for i in range(1, 7))))
    body += "<h2>关键配对比较</h2>" + selected.to_html(index=False, float_format="%.4f")
    body += "<h2>全部场景汇总</h2>" + aggregate.to_html(index=False, float_format="%.3f")
    (output / "结果总览.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>工程比较结果</title>'
        '<style>body{font:16px system-ui;margin:36px;color:#182a35}table{border-collapse:collapse;'
        'font-size:13px}td,th{border:1px solid #ccc;padding:8px}th{background:#e8f1f4}</style>'
        + body + "</html>", encoding="utf-8")
    print(json.dumps({"cases": len(specs), "runs": len(rows), "audit_failures": len(failures)},
                     ensure_ascii=False), flush=True)
    return frame, pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
