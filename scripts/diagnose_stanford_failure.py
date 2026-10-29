"""Describe already-inspected external failures without refitting or rotation rescoring."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json

from fogguard.evaluation.external_failure import (
    NAMES,
    distribution_table,
    event_opportunities,
    select_failures,
    summarize_windows,
    window_groups,
)
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_coordination import CoordinationFoG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-run", type=Path, default=Path("reports/a1_stanford_external_v1r1_20261029"))
    parser.add_argument("--train-reference", type=Path, default=Path("reports/a1_bilateral_v1_20261029"))
    parser.add_argument("--fogstar-raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, reference, output, root = args.external_run, args.train_reference, args.output_dir, Path(__file__).resolve().parents[1]
    if output.exists():
        parser.error("Use a new diagnosis directory")
    input_lock = json.loads((run / "INPUT_LOCK.json").read_text("utf-8"))
    evaluation = json.loads((run / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8"))
    check_lock(run, input_lock)
    check_lock(run / "evaluation", evaluation)
    check_lock(reference, json.loads((reference / "MODEL_LOCK.json").read_text("utf-8")))
    protocol = json.loads((reference / "protocol.json").read_text("utf-8"))
    if sha(args.fogstar_raw) != protocol["raw_sha256"]:
        raise ValueError("Source reference raw data changed")
    config = json.loads((run / "config.json").read_text("utf-8"))
    if sha(config["model"]) != config["model_sha256"]:
        raise ValueError("Model changed")
    predictions = pd.read_csv(run / "evaluation/predictions.csv", float_precision="round_trip")
    events = pd.read_csv(run / "evaluation/events.csv", float_precision="round_trip")
    opportunities = event_opportunities(events, predictions)
    split = json.loads((reference / "split.json").read_text("utf-8"))
    raw = pd.read_csv(args.fogstar_raw)
    train = raw.loc[raw.subjectID.isin(split["train"])].copy()
    del raw
    windows = anchored_gyro_windows(train, require_right=True)
    del train
    old = pd.read_csv(reference / "bilateral_rf/oof_predictions.csv", float_precision="round_trip")
    if len(windows) != len(old) or any(abs(w.end_s - r.window_end_s) > 1e-6 or int(w.subject_id) != r.subject_id for w, r in zip(windows, old.itertuples(), strict=True)):
        raise ValueError("TRAIN reference windows changed")
    np.testing.assert_array_equal([w.fog_label for w in windows], old.truth_fog_label)
    print(f"Describe TRAIN only: {len(windows)} windows; no source VAL/TEST constructed", flush=True)
    source = pd.DataFrame(summarize_windows(windows), columns=NAMES)
    source["source_person"] = [int(w.subject_id) for w in windows]
    source["truth_fog_fraction"] = [w.fog_fraction for w in windows]
    source["window_group"] = window_groups(source.truth_fog_fraction)
    source["saved_train_oof_score"] = old.fog_probability.to_numpy()
    del windows
    target_rows = []
    inventory = pd.read_csv(run / "input_inventory.csv")
    for item in inventory.itertuples():
        data = np.load(run / "inputs" / f"record_{item.record_id:03}.npz")
        frame = pd.DataFrame(data["gyro"], columns=DUAL)
        frame.insert(0, "timestamp", data["logical_time"])
        windows = anchored_gyro_windows(frame, require_right=True)
        saved = predictions.loc[predictions.record_id.eq(item.record_id)].reset_index(drop=True)
        if len(windows) != len(saved):
            raise ValueError("External schedule changed")
        np.testing.assert_allclose([w.end_s for w in windows], saved.window_end_s, atol=1e-6, rtol=0)
        part = pd.DataFrame(summarize_windows(windows), columns=NAMES)
        for field in ("window_start_s", "window_end_s", "available_at_s", "fog_probability", "truth_fog_fraction", "is_fog"):
            part[field] = saved[field].to_numpy()
        part["record_id"], part["source_person"] = item.record_id, item.source_person
        part["window_group"] = window_groups(part.truth_fog_fraction)
        target_rows.append(part)
    target = pd.concat(target_rows, ignore_index=True)
    output.mkdir(parents=True)
    source.to_csv(output / "train_invariant_summaries.csv", index=False)
    target.to_csv(output / "external_invariant_summaries.csv", index=False)
    distributions = distribution_table(source, target)
    distributions.to_csv(output / "pure_fog_distribution_shifts.csv", index=False)
    opportunities.to_csv(output / "event_opportunities.csv", index=False)
    counts = []
    for person, group in opportunities.groupby("source_person", sort=True):
        counts.append({"source_person": int(person), "events": len(group), "model_covered": int(group.covered.sum()),
            "label_reference_covered": int(group.label_reference_covered.sum()), "self_majority_reference_covered": int(group.self_majority_reference_covered.sum()),
            "missed_with_timely_self_majority": int(group.missed_despite_self_majority.sum()),
            "missed_without_timely_self_majority": int((~group.covered & group.timely_self_majority_windows.eq(0)).sum())})
    pd.DataFrame(counts).to_csv(output / "opportunity_by_person.csv", index=False)
    grouped = target.groupby(["source_person", "window_group"], observed=True).agg(windows=("fog_probability", "size"),
        median_score=("fog_probability", "median"), q90_score=("fog_probability", lambda x: x.quantile(0.9)),
        alarm_fraction=("is_fog", "mean")).reset_index()
    grouped.to_csv(output / "scores_by_person_purity.csv", index=False)
    model = CoordinationFoG.load(Path(config["model"]))
    changed = [name for name in model.feature_names if any(name.startswith(f"ankle{s}_gyro_{a}__") for s in ("L", "R") for a in ("x", "z")) and name.endswith(("__min", "__max", "__dynamic_median"))]
    importance = float(sum(v for n, v in zip(model.feature_names, model.classifier.feature_importances_, strict=True) if n in changed))
    summary = {"scope": "Post-hoc read-only external diagnostic, not new efficacy validation or model optimization",
        "external_records": len(inventory), "external_people": inventory.source_person.nunique(), "external_windows": len(target),
        "train_reference_windows": len(source), "train_pure_fog_windows": int(source.window_group.eq("pure_fog").sum()),
        "raw_events": len(opportunities), "original_model_covered": int(opportunities.covered.sum()),
        "label_reference_covered": int(opportunities.label_reference_covered.sum()),
        "self_majority_reference_covered": int(opportunities.self_majority_reference_covered.sum()),
        "missed_with_timely_self_majority": int(opportunities.missed_despite_self_majority.sum()),
        "missed_without_timely_self_majority": int((~opportunities.covered & opportunities.timely_self_majority_windows.eq(0)).sum()),
        "nominal_axis_map": [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], "per_trial_axis_alignment_verified": False,
        "nominal_sign_changed_feature_count": len(changed), "unchanged_feature_count": len(model.feature_names) - len(changed),
        "changed_features_train_impurity_importance_sum": importance,
        "rotated_target_model_scores_generated": 0, "thresholds_changed": 0, "models_fitted": 0,
        "label_reference_is_performance_ceiling": False}
    write_json(output / "summary.json", summary)
    write_json(output / "selected_failures.json", select_failures(opportunities))
    sources = [Path(__file__), root / "src/fogguard/evaluation/external_failure.py"]
    write_json(output / "DIAGNOSTIC_LOCK.json", {"external_input_lock_sha256": sha(run / "INPUT_LOCK.json"),
        "external_evaluation_sha256": sha(run / "evaluation/EVALUATION_COMPLETED.json"), "train_reference_lock_sha256": sha(reference / "MODEL_LOCK.json"),
        "model_sha256": sha(config["model"]), "source_sha256": {str(p.relative_to(root)): sha(p) for p in sources},
        "files": {str(p.relative_to(output)): sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
