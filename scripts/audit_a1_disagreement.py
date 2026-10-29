"""Audit saved model disagreements on continuous real validation events; no fitting."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json

from fogguard.evaluation.a1_disagreement import IDS, record_audit, segments, select_cases
from fogguard.evaluation.a1_event import EventSupport
from fogguard.evaluation.a1_fixed_split import role_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("reports/a1_coordination_v1_20261029"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, run, output = Path(__file__).resolve().parents[1], args.run_dir, args.output_dir
    if output.exists():
        parser.error("Use a new audit directory")
    protocol, split, lock = [json.loads((run / n).read_text("utf-8")) for n in ("protocol.json", "split.json", "MODEL_LOCK.json")]
    complete = json.loads((run / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8"))
    check_lock(run, lock)
    check_lock(run / "evaluation", complete)
    if sha(args.raw) != protocol["raw_sha256"] or complete["model_lock_sha256"] != sha(run / "MODEL_LOCK.json"):
        raise ValueError("Raw or lock changed")
    raw = role_frame(pd.read_csv(args.raw), split, "validation")
    names = ["bilateral_control", "coordination_rf"]
    frames = [pd.read_csv(run / "evaluation" / n / "predictions.csv", float_precision="round_trip") for n in names]
    keys = [*IDS, "window_start_s", "window_end_s"]
    np.testing.assert_allclose(frames[0][keys].astype(float), frames[1][keys].astype(float), atol=1e-6, rtol=0)
    records, events, pieces, payload = record_audit(raw, *frames)
    if records.shape[0] != raw.groupby(["subjectID", "sessionID", "taskID"]).ngroups:
        raise ValueError("Missing raw records")
    for label, frame, name in zip(("base", "candidate"), frames, names, strict=True):
        support = EventSupport.build(raw, frame, allowed=split["validation"])
        measured = support.metrics(frame.is_fog)
        np.testing.assert_allclose(records[label + "_true_s"].sum(), measured["true_alarm_s"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(records[label + "_false_s"].sum(), measured["false_alarm_s"], atol=1e-6, rtol=0)
        previous = pd.read_csv(run / "evaluation" / name / "event_coverage.csv")
        np.testing.assert_allclose(events[[*IDS, "start_s", "end_s"]], previous[[*IDS, "start_s", "end_s"]], atol=1e-6, rtol=0)
        np.testing.assert_allclose(events[label + "_alarm_s"], previous.alarm_overlap_s, atol=1e-6, rtol=0)
        np.testing.assert_allclose(events.support_s, previous.model_support_s, atol=1e-6, rtol=0)
        np.testing.assert_allclose(events[label + "_first_overlap_delay_s"], previous.onset_delay_s, atol=1e-6, rtol=0)
    cases = select_cases(events, pieces)
    output.mkdir(parents=True)
    records.to_csv(output / "records.csv", index=False)
    events.to_csv(output / "events.csv", index=False)
    pieces.to_csv(output / "disagreement_segments.csv", index=False)
    changed = frames[0].is_fog.ne(frames[1].is_fog)
    windows = frames[0].loc[changed, [*keys, "truth_fog_label", "fog_probability", "is_fog"]].copy()
    windows = windows.rename(columns={"fog_probability": "base_probability", "is_fog": "base_is_fog"})
    windows["candidate_probability"] = frames[1].loc[changed, "fog_probability"]
    windows["candidate_is_fog"] = frames[1].loc[changed, "is_fog"]
    windows.to_csv(output / "window_disagreements.csv", index=False)
    write_json(output / "selected_cases.json", cases)
    summary = {"scope": "Readonly already-inspected VALIDATION4; no new predictions, refit or threshold change",
        "raw_records": len(records), "raw_samples": len(raw), "truth_events": len(events), "matched_windows": len(frames[0]),
        "window_disagreements": int(changed.sum()), "base_true_s": float(records.base_true_s.sum()),
        "candidate_true_s": float(records.candidate_true_s.sum()), "base_false_s": float(records.base_false_s.sum()),
        "candidate_false_s": float(records.candidate_false_s.sum()),
        **{name: float(records[name].sum()) for name in ("lost_fog_s", "recovered_fog_s", "removed_false_alarm_s", "new_false_alarm_s")},
        "lost_partition_s": {part: float(events[part + "_lost_s"].sum()) for part in ("onset", "interior", "ending")},
        "base_events_any_overlap": int(events.base_covered.sum()), "candidate_events_any_overlap": int(events.candidate_covered.sum()),
        "source_comparison_and_event_tables_match": True, "new_test_windows_constructed": 0}
    write_json(output / "summary.json", summary)
    # No raw sample or decision is downsampled for metrics. Only display signals use 15Hz.
    display = []
    for fields, signal, prediction_frames, states in payload:
        t = signal.timestamp.to_numpy(float)
        channels = [[f"ankle{s}_gyro_{a}" for a in "xyz"] for s in ("L", "R")]
        mag = [np.sqrt(np.square(signal[c].to_numpy(float)).sum(axis=1)) for c in channels]
        record = {"key": "/".join(str(fields[k]) for k in IDS), "ids": fields, "start": t[0], "stop": t[-1] + 1 / 60,
                  "signals": [[round(float(t[i]), 4), *[round(float(m[i]), 3) if np.isfinite(m[i]) else None for m in mag]] for i in range(0, len(t), 4)],
                  "truth": [[float(t[a]), float(t[b - 1] + 1 / 60)] for a, b in segments(t, signal.fog.to_numpy(bool))]}
        for label, frame, state in zip(("base", "candidate"), prediction_frames, states, strict=True):
            record[label] = {"threshold": float(frame.fog_threshold.iloc[0]) if len(frame) else None,
                "decisions": [[float(r.window_end_s), min(float(r.window_end_s + 0.5), float(frame.window_end_s.iloc[i + 1]) if i + 1 < len(frame) else record["stop"], record["stop"]), float(r.fog_probability)] for i, r in enumerate(frame.itertuples())],
                "support": [[float(t[a]), float(t[b - 1] + 1 / 60)] for a, b in segments(t, state >= 0)],
                "alarms": [[float(t[a]), float(t[b - 1] + 1 / 60)] for a, b in segments(t, state == 1)]}
        display.append(record)
    data = {"records": display, "cases": cases, "events": json.loads(events.to_json(orient="records")), "summary": summary}
    write_json(output / "replay_data.json", data)
    template = root / "src/fogguard/presentation/a1_disagreement_review.html"
    embedded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    (output / "index.html").write_text(template.read_text("utf-8").replace("__REPLAY_DATA__", embedded), encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-disagreement-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir()
    for case in cases:
        record = next(r for r in display if all(r["ids"][k] == case[k] for k in IDS))
        start, stop = max(record["start"], case["start_s"] - 4), min(record["stop"], case["end_s"] + 4)
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.1, top=0.86, hspace=0.25)
        points = np.array([[t, l if l is not None else np.nan, r if r is not None else np.nan] for t, l, r in record["signals"]])
        visible = (points[:, 0] >= start) & (points[:, 0] <= stop)
        axes[0].plot(points[visible, 0], points[visible, 1], lw=0.7, label="Left gyro magnitude")
        axes[0].plot(points[visible, 0], points[visible, 2], lw=0.7, label="Right gyro magnitude")
        axes[0].set_ylabel("deg/s\n(display 15Hz)")
        for label, color in (("base", "#287da0"), ("candidate", "#c07832")):
            for a, b, p in record[label]["decisions"]:
                if b > start and a < stop:
                    axes[1].plot([a, b], [p, p], color=color, lw=1)
            axes[1].axhline(record[label]["threshold"], color=color, linestyle="--", lw=0.8, label=f"{label} threshold {record[label]['threshold']:.2f}")
            row = 1 if label == "base" else 0
            axes[2].broken_barh([(a, b - a) for a, b in record[label]["support"]], (row - 0.13, 0.26), facecolors="#e6e6e6")
            axes[2].broken_barh([(a, b - a) for a, b in record[label]["alarms"]], (row - 0.13, 0.26), facecolors=color)
        axes[2].broken_barh([(a, b - a) for a, b in record["truth"]], (1.87, 0.26), facecolors="#8c7aa5")
        for ax in axes:
            for a, b in record["truth"]:
                ax.axvspan(a, b, color="#8c7aa5", alpha=0.08)
            ax.set_xlim(start, stop)
            ax.grid(alpha=0.15)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[1].set(ylim=(0, 1), ylabel="FoG score")
        axes[1].legend(fontsize=8)
        axes[2].set(yticks=[0, 1, 2], yticklabels=["78D alarm", "66D alarm", "Raw FoG"], xlabel="Raw record time (s)", ylim=(-0.5, 2.5))
        fig.suptitle(f"{case['case']} | P/S/T {record['key']}\nLocked VALIDATION replay; gray = supported non-alarm, blank = unknown; not new test", fontsize=11)
        for suffix in ("png", "svg"):
            fig.savefig(figures / f"{case['case']}.{suffix}", dpi=180)
        plt.close(fig)
    source_paths = [Path(__file__), root / "src/fogguard/evaluation/a1_disagreement.py", template]
    write_json(output / "AUDIT_COMPLETED.json", {"kind": "Readonly saved-prediction audit; no new model scores",
        "source_model_lock_sha256": sha(run / "MODEL_LOCK.json"), "source_evaluation_sha256": sha(run / "evaluation/EVALUATION_COMPLETED.json"),
        "raw_sha256": sha(args.raw), "source_sha256": {str(p.relative_to(root)): sha(p) for p in source_paths},
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
