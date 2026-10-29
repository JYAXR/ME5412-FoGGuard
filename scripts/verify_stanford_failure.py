"""Independently recompute diagnostic descriptors and per-event temporal opportunities."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json
from scipy.signal import butter, sosfiltfilt

from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows


def direct_descriptors(samples):
    """Separate per-ankle, per-axis calculation, without diagnostic helper imports."""
    result = []
    for ankle in (samples[:, :3], samples[:, 3:]):
        filtered = np.column_stack([sosfiltfilt(butter(4, (0.5, 20), fs=60, btype="bandpass", output="sos"), ankle[:, a]) for a in range(3)])
        energy = sum(abs(np.fft.rfft(filtered[:, a])) ** 2 for a in range(3)) / 14400
        gait, freeze = energy[1:6].sum(), energy[6:17].sum()
        result.extend([np.sqrt((filtered**2).sum() / 120), gait, freeze, freeze / max(gait, 1e-12),
                       0.5 * (energy[1:6].argmax() + 1) if gait > 1e-12 else 0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--external-run", type=Path, default=Path("reports/a1_stanford_external_v1r1_20261029"))
    parser.add_argument("--train-reference", type=Path, default=Path("reports/a1_bilateral_v1_20261029"))
    parser.add_argument("--fogstar-raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, external, reference = args.run_dir, args.external_run, args.train_reference
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((run / "DIAGNOSTIC_LOCK.json").read_text("utf-8"))
    check_lock(run, lock)
    for path, digest in lock["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError("Diagnostic code changed")
    for path, key in ((external / "INPUT_LOCK.json", "external_input_lock_sha256"),
                      (external / "evaluation/EVALUATION_COMPLETED.json", "external_evaluation_sha256"),
                      (reference / "MODEL_LOCK.json", "train_reference_lock_sha256")):
        if sha(path) != lock[key]:
            raise ValueError("Upstream lock changed")
        check_lock(path.parent, json.loads(path.read_text("utf-8")))
    config = json.loads((external / "config.json").read_text("utf-8"))
    if sha(config["model"]) != lock["model_sha256"]:
        raise ValueError("Frozen model changed")
    predictions = pd.read_csv(external / "evaluation/predictions.csv", float_precision="round_trip")
    events = pd.read_csv(external / "evaluation/events.csv", float_precision="round_trip")
    diagnosed = pd.read_csv(run / "event_opportunities.csv", float_precision="round_trip")
    assert len(events) == len(diagnosed) == 211
    for original, row in zip(events.itertuples(), diagnosed.itertuples(), strict=True):
        assert (original.record_id, original.start_s, original.end_s) == (row.record_id, row.start_s, row.end_s)
        frame = predictions.loc[predictions.record_id.eq(row.record_id)].sort_values("available_at_s")
        tally = {"timely": 0, "weak": 0, "own": 0, "actual": 0.0, "label": 0.0}
        score, weak_score = [], []
        records = list(frame.itertuples())
        for i, p in enumerate(records):
            end = min(p.available_at_s + 0.5, records[i + 1].available_at_s if i + 1 < len(records) else float("inf"))
            duration = max(0, min(end, row.end_s) - max(p.available_at_s, row.start_s))
            own = sum(row.start_s - 1e-9 <= p.window_start_s + j / 60 < row.end_s - 1e-9 for j in range(120)) >= 60
            if duration > 1e-6:
                tally["timely"] += 1
                tally["weak"] += int(p.truth_fog_label)
                tally["own"] += int(own)
                score.append(p.fog_probability)
                if p.truth_fog_label:
                    weak_score.append(p.fog_probability)
            if p.is_fog:
                tally["actual"] += duration
            if p.truth_fog_label:
                tally["label"] += duration
        assert (tally["timely"], tally["weak"], tally["own"]) == (row.timely_windows, row.timely_weak_positive_windows, row.timely_self_majority_windows)
        np.testing.assert_allclose([tally["actual"], tally["label"]], [row.alarm_s, row.label_reference_overlap_s], atol=1e-8, rtol=0)
        assert bool(tally["weak"]) == row.label_reference_covered
        assert bool(tally["own"]) == row.self_majority_reference_covered
        assert (not row.covered and bool(tally["own"])) == row.missed_despite_self_majority
        np.testing.assert_allclose([max(score, default=np.nan), max(weak_score, default=np.nan)],
                                   [row.max_timely_score, row.max_timely_weak_positive_score], equal_nan=True)
    source = pd.read_csv(run / "train_invariant_summaries.csv", float_precision="round_trip")
    target = pd.read_csv(run / "external_invariant_summaries.csv", float_precision="round_trip")
    names = [f"{side}_{field}" for side in ("left", "right") for field in ("vector_rms", "gait_power", "freeze_power", "freeze_ratio", "gait_peak_hz")]
    for field in ("record_id", "window_start_s", "window_end_s", "available_at_s", "fog_probability", "truth_fog_fraction", "is_fog"):
        np.testing.assert_array_equal(target[field], predictions[field])
    for item in pd.read_csv(external / "input_inventory.csv").itertuples():
        data = np.load(external / f"inputs/record_{item.record_id:03}.npz")
        frame = pd.DataFrame(data["gyro"], columns=DUAL)
        frame.insert(0, "timestamp", data["logical_time"])
        windows = anchored_gyro_windows(frame, require_right=True)
        actual = target.loc[target.record_id.eq(item.record_id), names].to_numpy()
        np.testing.assert_allclose([direct_descriptors(w.samples) for w in windows], actual, atol=1e-8, rtol=1e-10)
    protocol = json.loads((reference / "protocol.json").read_text("utf-8"))
    assert sha(args.fogstar_raw) == protocol["raw_sha256"]
    train_ids = json.loads((reference / "split.json").read_text("utf-8"))["train"]
    raw = pd.read_csv(args.fogstar_raw)
    windows = anchored_gyro_windows(raw.loc[raw.subjectID.isin(train_ids)], require_right=True)
    assert len(windows) == len(source) == 6383
    np.testing.assert_array_equal(source.source_person, [int(w.subject_id) for w in windows])
    np.testing.assert_allclose(source.truth_fog_fraction, [w.fog_fraction for w in windows], atol=1e-12)
    np.testing.assert_allclose([direct_descriptors(w.samples) for w in windows], source[names], atol=1e-8, rtol=1e-10)
    old_oof = pd.read_csv(reference / "bilateral_rf/oof_predictions.csv", float_precision="round_trip")
    np.testing.assert_array_equal(source.saved_train_oof_score, old_oof.fog_probability)
    for frame in (source, target):
        expected = ["pure_fog" if x >= 1 - 1e-12 else "pure_nonfog" if x <= 1e-12 else "mixed" for x in frame.truth_fog_fraction]
        np.testing.assert_array_equal(frame.window_group, expected)
    selected = json.loads((run / "selected_failures.json").read_text("utf-8"))
    for selected_row in selected:
        candidates = diagnosed.loc[diagnosed.source_person.eq(selected_row["source_person"]) & ~diagnosed.covered & diagnosed.timely_self_majority_windows.gt(0)]
        winner = candidates.sort_values(["duration_s", "record_id", "start_s"], ascending=[False, True, True]).iloc[0]
        assert (selected_row["record_id"], selected_row["start_s"]) == (winner.record_id, winner.start_s)
    write_json(run / "verification.json", {"status": "PASS", "raw_events_recomputed": len(events),
        "external_windows_independently_recomputed": len(target), "train_windows_independently_recomputed": len(source),
        "diagnostic_lock_sha256": sha(run / "DIAGNOSTIC_LOCK.json"), "verifier_sha256": sha(Path(__file__)),
        "models_fitted": 0, "new_model_scores": 0})
    print("PASS: 211 event opportunities, 10033 external + 6383 TRAIN descriptors; unchanged frozen inputs/scores/model")


if __name__ == "__main__":
    main()
