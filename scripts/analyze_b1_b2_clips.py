"""Summarize the existing joint replay and explore unverified ankle motion peaks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fogguard.motion.candidate_peaks import candidate_motion_peaks

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(plan, joint, output):
    if output.exists():
        raise ValueError("Choose a fresh output directory to preserve previous results")
    manifest = json.loads((joint / "MANIFEST.json").read_text(encoding="utf-8"))["files_sha256"]
    for name, digest in manifest.items():
        if sha(joint/name) != digest:
            raise ValueError(f"Joint output changed: {name}")
    frozen = json.loads((plan / "PLAN.json").read_text(encoding="utf-8"))
    for name, digest in frozen["csv_sha256"].items():
        if sha(plan/name) != digest:
            raise ValueError(f"Sample changed: {name}")
    if sha(plan/"PLAN.json") != json.loads((joint/"RESULTS.json").read_text(encoding="utf-8"))["sampling_plan_sha256"]:
        raise ValueError("Joint replay and sample plan differ")
    output.mkdir(parents=True)
    cases = pd.read_csv(plan/"sample_manifest.csv")
    details, events, transitions, activity_samples = [], [], [], []
    for case in cases["clip"]:
        sensor = pd.read_csv(plan/"clips"/f"{case}_sensors.csv")
        truth = pd.read_csv(plan/"clips"/f"{case}_truth.csv")
        if not (sensor.timestamp.to_numpy() == truth.timestamp.to_numpy()).all():
            raise ValueError("Sensor and truth clocks are not aligned")
        # Candidate algorithm only receives sensor data, never labels.
        peaks = candidate_motion_peaks(sensor)
        permissive = candidate_motion_peaks(sensor, height=15, prominence=8)
        strict = candidate_motion_peaks(sensor, height=30, prominence=15)
        if not peaks.equals(candidate_motion_peaks(sensor.assign(fog=1, activity=7))):
            raise ValueError("Candidate peaks depend on injected labels")
        origin = float(sensor.timestamp.iloc[0])
        for peak in peaks.itertuples(index=False):
            sample = (sensor.timestamp-float(peak.timestamp_s)).abs().idxmin()
            events.append({"case":case,"time_from_clip_start_s":float(peak.timestamp_s)-origin,
                           "side":peak.side,"angular_speed_deg_s":peak.angular_speed_deg_s,
                           "activity_for_review_only":int(truth.activity.iloc[sample]),
                           "fog_for_review_only":int(truth.fog.iloc[sample])})
        for activity, group in truth.groupby("activity"):
            activity_samples.append({"case":case,"activity":int(activity),"samples":len(group)})
        folder = joint/"real"/case
        perception = pd.read_csv(folder/"perception_runtime.csv")
        feedback = pd.read_csv(folder/"feedback_runtime.csv")
        cmd = pd.read_csv(folder/"commands_march.csv")
        fixed = pd.read_csv(folder/"commands_fixed.csv")
        for row in cmd.loc[cmd.guidance_action.isin(["start","stop"])].itertuples(index=False):
            transitions.append({"case":case,"time_from_clip_start_s":float(row.guidance_timestamp_s)-origin,
                                "action":row.guidance_action,"reason":row.guidance_reason,
                                "turn":row.turn,"motion_feedback":row.proxy_feedback_state})
        details.append({"case":case,"sensor_seconds":len(sensor)/60,
                        "annotated_fog_seconds":float(truth.fog.eq(1).sum()/60),
                        "left_motion_peaks":int(peaks.side.eq("left").sum()),
                        "right_motion_peaks":int(peaks.side.eq("right").sum()),
                        "permissive_motion_peaks":len(permissive),
                        "strict_motion_peaks":len(strict),
                        "confirmed_rows":int(perception.event_phase.eq("confirmed").sum()),
                        "direction_unknown_rows":int(perception.turn.eq("unknown").sum()),
                        "feedback_unknown_windows":int(feedback.feedback_state.eq("unknown").sum()),
                        "fixed_starts":int(fixed.guidance_action.eq("start").sum()),
                        "march_starts":int(cmd.guidance_action.eq("start").sum()),
                        "march_stop_reasons":json.dumps(cmd.loc[cmd.guidance_action.eq("stop"),"guidance_reason"].value_counts().to_dict())})
    d=pd.DataFrame(details); e=pd.DataFrame(events); a=pd.DataFrame(activity_samples)
    d.to_csv(output/"clip_summary.csv",index=False)
    e.to_csv(output/"candidate_motion_peaks.csv",index=False)
    pd.DataFrame(transitions).to_csv(output/"march_transitions.csv",index=False)
    exposure=a.groupby("activity",as_index=False).samples.sum()
    exposure["seconds"]=exposure.samples/60
    exposure["candidate_peaks"]=exposure.activity.map(e.activity_for_review_only.value_counts()).fillna(0).astype(int)
    exposure.to_csv(output/"activity_exploration.csv",index=False)
    report={"clips":len(d),"candidate_motion_peaks":len(e),"left":int(e.side.eq("left").sum()),
            "right":int(e.side.eq("right").sum()),"with_march_start":int(d.march_starts.gt(0).sum()),
            "sensitivity_peak_counts":{"15_deg_s_prominence_8":int(d.permissive_motion_peaks.sum()),
                                       "20_deg_s_prominence_10":len(e),
                                       "30_deg_s_prominence_15":int(d.strict_motion_peaks.sum())},
            "with_no_march_start":int(d.march_starts.eq(0).sum()),"source_files":{
                "candidate_peaks.py":sha(ROOT/"src/fogguard/motion/candidate_peaks.py"),
                "analyze_b1_b2_clips.py":sha(ROOT/"scripts/analyze_b1_b2_clips.py")},
            "plan_sha256":sha(plan/"PLAN.json"),"joint_manifest_sha256":sha(joint/"MANIFEST.json"),
            "limitations":["Offline unvalidated angular-speed peaks; not observed footfalls",
                           "Activity and FoG labels are for post-hoc review only",
                           "Validation data already used in development; no independent step truth",
                           "Candidate timestamps are not real-time and are not fed to the controller"]}
    (output/"analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--joint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.plan,args.joint,args.output)
