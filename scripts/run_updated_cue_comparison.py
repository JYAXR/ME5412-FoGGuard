"""Updated cue comparison on frozen real-model outputs and separate rule fixtures."""
from __future__ import annotations

import argparse
import itertools
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from run_b1_b2_joint_review import _check_inputs, _clock

from fogguard.cueing.contracts import GuidanceConfig
from fogguard.cueing.proxy_replay import replay_proxy_guidance
from fogguard.evaluation.automation2 import audit_commands
from fogguard.evaluation.randomized_plan import file_hash
from fogguard.evaluation.randomized_receiver import receiver_issues
from fogguard.motion.proxy_feedback import GYRO, MotionProxyConfig, estimate_motion_feedback
from fogguard.perception import fuse_perception

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("fixed", "feedback", "countdown", "march", "no_fade")
PROFILES = ("low_motion", "recovery_early", "recovery_late", "feedback_gap",
            "unknown_direction", "uncertain_motion", "normal", "refreeze")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf8")


def fixture(profile, turn):
    """Rule probes: constructed source candidates and constructed sensor signals."""
    ends = np.arange(2, 22, .5)
    scores = np.full(len(ends), .9)
    if profile == "normal":
        scores[:] = .1
    elif profile in {"recovery_early", "recovery_late"}:
        scores[ends >= (8 if profile == "recovery_early" else 12)] = .1
    elif profile == "refreeze":
        scores[(ends >= 8) & (ends < 8.5)] = .1
    common = {"subject_id": "fixture", "session_id": "1", "task_id": "1",
              "window_start_s": ends - 119/60, "window_end_s": ends,
              "available_at_s": ends, "model_version": "constructed-rule-v1",
              "prediction_source": "constructed_rule_fixture"}
    fog = pd.DataFrame(common).assign(fog_probability=scores, fog_threshold=.5, is_fog=scores >= .5)
    direction = pd.DataFrame(common).assign(direction=turn, direction_confidence=.9,
                                           direction_task="current", horizon_s=0., input_valid=True)
    if profile == "unknown_direction":
        direction.loc[(ends >= 6) & (ends < 8), "direction"] = "unknown"
    times = np.arange(22*60)/60
    sensor = pd.DataFrame({"timestamp": times, "subjectID": "fixture", "sessionID": "1", "taskID": "1"})
    for channel in GYRO:
        sensor[channel] = 0.
    if profile in {"recovery_early", "recovery_late", "normal", "refreeze"}:
        sensor["ankleL_gyro_x"] = 20.
        sensor["ankleR_gyro_x"] = 20.
    if profile == "uncertain_motion":
        sensor["ankleL_gyro_x"] = 8.
        sensor["ankleR_gyro_x"] = 8.
    if profile == "feedback_gap":
        sensor.loc[(times >= 5) & (times < 7.5), "ankleL_gyro_x"] = np.nan
    return sensor, fog, direction


def duration_vectors(commands):
    frame = commands.loc[~commands.guidance_terminal].copy()
    times = frame.guidance_timestamp_s.to_numpy(float)
    delta = np.diff(times, append=times[-1])
    if (delta < 0).any():
        raise ValueError("Command clock reversed")
    return frame, delta


def measure(commands, strategy, truth=None):
    frame, delta = duration_vectors(commands)
    on = frame.guidance_visual_enabled.to_numpy(bool)
    unsafe = frame.guidance_visual_enabled & (~frame.input_fresh | frame.turn.eq("unknown"))
    if strategy != "fixed":
        unsafe |= frame.guidance_visual_enabled & ~frame.proxy_feedback_valid_now
    nonfog_s, after_last_fog_s = None, None
    if truth is not None:
        # Evaluation only, never included in controller inputs.
        t = truth.timestamp.to_numpy(float)
        idx = np.searchsorted(t, frame.guidance_timestamp_s.to_numpy(float), side="right")-1
        valid = (idx >= 0) & (idx < len(t))
        y = truth.fog.to_numpy(int)[np.clip(idx, 0, len(t)-1)]
        nonfog_s = float(np.sum(delta * on * valid * (y == 0)))
        positive = np.flatnonzero(truth.fog.to_numpy(int) == 1)
        if len(positive) and positive[-1]+1 < len(truth):
            last_end = float(t[positive[-1]+1])
            after_last_fog_s = float(np.sum(delta * on * (frame.guidance_timestamp_s.to_numpy(float) >= last_end)))
    audit = audit_commands(commands)
    spacing = frame.guidance_step_length_m
    return {"starts":int(frame.guidance_action.eq("start").sum()),
            "stops":int(commands.guidance_action.eq("stop").sum()),
            "active_s":float(np.sum(delta*on)),
            "visual_intensity_s":float(np.sum(delta*on*frame.guidance_visual_opacity.to_numpy(float))),
            "audio_volume_s":float(np.sum(delta*frame.guidance_audio_enabled.to_numpy(bool)*frame.guidance_audio_volume.to_numpy(float))),
            "fade_entries":int(frame.guidance_reason.eq("perception_recovery_with_motion_proxy_begin_fade").sum()),
            "fallback_entries":int(frame.guidance_reason.isin(["valid_no_motion_enter_march", "valid_no_motion_enter_countdown"]).sum()),
            "fallback_timeout_returns":int(frame.guidance_reason.eq("fallback_timeout_return_forward_unverified").sum()),
            "post_last_fog_active_s":after_last_fog_s, "nonfog_active_s":nonfog_s,
            "unclosed_commands":len(frame.loc[frame.guidance_action.eq("start")])-len(commands.loc[commands.guidance_action.eq("stop")]),
            "lifecycle_errors":len(audit), "unsafe_active_rows":int(unsafe.sum()),
            "actual_steps_claimed":int(frame.guidance_stable_steps.ne(0).sum()),
            "spacing_outside_range_rows":int((~spacing.between(.32-1e-9,.55+1e-9)).sum()),
            "measured_spacing_error_m":None,
            "stop_reasons":json.dumps(commands.loc[commands.guidance_action.eq("stop"), "guidance_reason"].value_counts().to_dict(), sort_keys=True)}


def one_case(sensor, fog, direction, config, folder, *, case, origin, truth=None):
    if any({"fog", "activity", "fog_severity"} & set(f) or any(str(c).startswith("truth_") for c in f)
           for f in (sensor, fog, direction)):
        raise ValueError("Runtime inputs contain labels")
    fog = fog.copy()
    if "input_valid" not in fog:
        fog["input_valid"] = True
    feedback = estimate_motion_feedback(sensor, config)
    clock = _clock(sensor, feedback)
    p = fuse_perception(fog, direction, clock=clock)
    p = p.loc[p.available_at_s.le(float(sensor.timestamp.iloc[-1])+1e-8)].reset_index(drop=True)
    p["cue_test_input_origin"] = origin
    checks = receiver_issues(fog, direction, p)
    if checks:
        raise ValueError(f"Perception receiver issues: {checks[:2]}")
    folder.mkdir(parents=True)
    p.to_csv(folder/"perception_runtime.csv",index=False)
    feedback.to_csv(folder/"feedback_runtime.csv",index=False)
    records = []
    for name in STRATEGIES:
        config_g = replace(GuidanceConfig(), enable_fading=name != "no_fade")
        commands = replay_proxy_guidance(p, feedback, strategy="march" if name == "no_fade" else name,
                                         proxy_config=config, guidance_config=config_g, scenario_id=case)
        commands.to_csv(folder/f"commands_{name}.csv", index=False)
        records.append({"case":case,"origin":origin,"strategy":name,
                        **measure(commands,name,truth),"receiver_errors":len(checks)})
    return records


def run(plan, model_run, package, prior_joint, output):
    if output.exists():
        raise ValueError("Choose a fresh output directory")
    _frozen, model_result = _check_inputs(plan,model_run,package)
    prior = json.loads((prior_joint/"RESULTS.json").read_text("utf8"))
    if prior["sampling_plan_sha256"] != file_hash(plan/"PLAN.json"):
        raise ValueError("Source plan differs from previous locked replay")
    config = MotionProxyConfig.load(ROOT/"config/b1_motion_feedback.json")
    samples = pd.read_csv(plan/"sample_manifest.csv")
    output.mkdir(parents=True)
    rows=[]
    for i, sample in enumerate(samples.itertuples(index=False)):
        paths = {"sensor":plan/"clips"/f"{sample.clip}_sensors.csv",
                 "fog":model_run/"runtime"/f"{sample.clip}_clean_bilateral_rf.csv",
                 "direction":model_run/"runtime"/f"{sample.clip}_clean_current_direction.csv",
                 "truth_evaluation_only":plan/"clips"/f"{sample.clip}_truth.csv"}
        for name,path in paths.items():
            if prior["input_sha256"][f"{sample.clip}/{name}"] != file_hash(path):
                raise ValueError(f"Source input changed: {sample.clip} {name}")
        frames={name:pd.read_csv(path) for name,path in paths.items()}
        rows.extend(one_case(frames["sensor"],frames["fog"],frames["direction"],config,
                             output/"real"/sample.clip,case=sample.clip,origin="real_model_outputs_and_imu_proxy",
                             truth=frames["truth_evaluation_only"]))
        if (i+1)%4==0:
            print(f"Updated real replay: {i+1}/{len(samples)}",flush=True)
    for profile,turn in itertools.product(PROFILES,("straight","left","right")):
        sensor,fog,direction=fixture(profile,turn)
        case=f"{profile}_{turn}"
        rows.extend(one_case(sensor,fog,direction,config,output/"constructed"/case,
                             case=case,origin="constructed_rule_fixture"))
    table=pd.DataFrame(rows)
    table.to_csv(output/"runs.csv",index=False)
    sums=["starts","stops","active_s","visual_intensity_s","audio_volume_s","fade_entries",
          "fallback_entries","fallback_timeout_returns","lifecycle_errors","unsafe_active_rows",
          "actual_steps_claimed","spacing_outside_range_rows","receiver_errors"]
    agg={key:(key,"sum") for key in sums}
    summary=table.groupby(["origin","strategy"],as_index=False).agg(cases=("case","count"),**agg)
    summary.to_csv(output/"summary.csv",index=False)
    pairs=[]
    for origin,group in table.groupby("origin"):
        a=group.loc[group.strategy.eq("march")].set_index("case")
        for other in ("no_fade","feedback","countdown","fixed"):
            b=group.loc[group.strategy.eq(other)].set_index("case").reindex(a.index)
            for metric in ("active_s","visual_intensity_s","audio_volume_s","starts","fallback_entries"):
                diff=a[metric]-b[metric]
                pairs.append({"origin":origin,"reference":"march","alternative":other,"metric":metric,
                              "pairs":len(diff),"reference_mean":float(a[metric].mean()),
                              "alternative_mean":float(b[metric].mean()),"mean_difference":float(diff.mean()),
                              "lower":int(diff.lt(-1e-8).sum()),"equal":int(diff.abs().le(1e-8).sum()),"higher":int(diff.gt(1e-8).sum())})
    pd.DataFrame(pairs).to_csv(output/"paired_comparisons.csv",index=False)
    source_names=["scripts/run_updated_cue_comparison.py","scripts/run_b1_b2_joint_review.py",
                  "src/fogguard/cueing/proxy_replay.py","src/fogguard/motion/proxy_feedback.py",
                  "config/b1_motion_feedback.json","src/fogguard/perception.py",
                  "src/fogguard/context/fogstar.py","src/fogguard/cueing/guidance.py",
                  "src/fogguard/cueing/contracts.py","src/fogguard/cueing/targets.py",
                  "src/fogguard/evaluation/automation2.py","src/fogguard/evaluation/randomized_receiver.py"]
    failures=table[["lifecycle_errors","unsafe_active_rows","actual_steps_claimed","spacing_outside_range_rows","receiver_errors"]].sum().sum()
    result={"schema":"fogguard.b1_b2.updated_cue_comparison.v1","status":"PASS" if failures==0 else "FAIL",
            "real_cases":len(samples),"constructed_cases":len(PROFILES)*3,"strategies":list(STRATEGIES),"runs":len(table),
            "model_hashes":model_result["model_hashes"],"plan_sha256":file_hash(plan/"PLAN.json"),
            "prior_joint_manifest_sha256":file_hash(prior_joint/"MANIFEST.json"),
            "source_sha256":{name:file_hash(ROOT/name) for name in source_names},
            "input_sha256":prior["input_sha256"],"python":platform.python_version(),
            "motion_config":config.as_dict(),"base_guidance_config":GuidanceConfig().as_dict(),
            "summary":json.loads(summary.to_json(orient="records")),
            "limits":["Real inputs use frozen new-model predictions; constructed rule probes do not run learned models",
                      "Real data is previously inspected development validation; not a new TEST",
                      "Motion feedback is an uncalibrated movement-only proxy; no measured footfalls or positions",
                      "Spacing stays at the configured value; measured spacing accuracy is unavailable",
                      "Intensity integral is a software exposure measure, not patient benefit",
                      "No model retuning or threshold change; same inputs for every strategy"]}
    write_json(output/"RESULTS.json",result)
    write_json(output/"MANIFEST.json",{"files_sha256":{p.relative_to(output).as_posix():file_hash(p)
               for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({"status":result["status"],"real_cases":len(samples),"constructed_cases":len(PROFILES)*3,"runs":len(table)}))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("plan","model-run","package","prior-joint","output"):
        parser.add_argument("--"+flag,type=Path,required=True)
    args=parser.parse_args()
    result=run(args.plan,args.model_run,args.package,args.prior_joint,args.output)
    raise SystemExit(result["status"]!="PASS")

