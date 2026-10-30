"""Recount paired metrics, replay saved forests and audit personalization splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

KEYS = ["subject_id", "session_id", "task_id"]
STRATEGIES = ["population", "personal", "population_calibrated"]


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask(meta, identities):
    keys = set(map(tuple, identities))
    return np.array([tuple(row) in keys for row in meta[KEYS].itertuples(index=False, name=None)])


def counts(frame):
    y, p = frame.truth_fog_label.to_numpy(int), frame.is_fog.to_numpy(bool)
    tp, fp = int(((y == 1) & p).sum()), int(((y == 0) & p).sum())
    fn, tn = int(((y == 1) & ~p).sum()), int(((y == 0) & ~p).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fog_f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh verification output")
    root = Path(__file__).resolve().parents[1]
    lock, result = read(args.run / "PLAN_LOCK.json"), read(args.run / "results.json")
    manifest = read(args.run / "MANIFEST.json")
    assert digest(args.raw) == lock["raw_sha256"]
    for name, expected in manifest["files"].items():
        assert digest(args.run / name) == expected, name
    for name, expected in lock["source_sha256"].items():
        assert digest(root / name) == expected, name
    config = lock["config"]
    base = read(root / "config/fog_selected.json")
    people = set(map(str, base["split"]["train"]))
    meta = pd.read_csv(args.run / "window_metadata.csv", dtype={k: str for k in KEYS})
    x = np.load(args.run / "features.npz", allow_pickle=False)["values"]
    assert set(meta.subject_id) == people and x.shape == (len(meta), 66)
    assert not people & set(map(str, base["split"]["validation"] + base["split"]["test"]))
    frames = {s: pd.read_csv(args.run / f"{s}_evaluation.csv", dtype={k: str for k in KEYS})
              for s in STRATEGIES}
    folds = read(args.run / "folds.json")
    replayed = 0
    for fold in folds:
        adapt, held = mask(meta, fold["adaptation_records"]), mask(meta, fold["evaluation_records"])
        assert not np.any(adapt & held)
        if fold["split_kind"] == "session_held_out":
            assert not {r[1] for r in fold["adaptation_records"]} & {r[1] for r in fold["evaluation_records"]}
        positives = int(meta.loc[adapt, "truth_fog_label"].sum())
        negatives = int(adapt.sum()) - positives
        expected_eligible = bool(held.any() and positives >= config["minimum_adaptation_positive_windows"]
                                 and negatives >= config["minimum_adaptation_negative_windows"])
        assert expected_eligible == fold["eligible"]
        if not expected_eligible:
            continue
        target = fold["subject_id"]
        population = joblib.load(args.run / "models" / f"population_p{target}.joblib")
        personal = joblib.load(args.run / "models" / fold["fold_id"] / "personal.joblib")
        assert set(population.fit_participants) == people - {target}
        assert personal.fit_participants == (target,)
        assert set(personal.adaptation_records) == set(map(tuple, fold["adaptation_records"]))
        pscore = population.classifier.predict_proba(x[held])[:, list(population.classifier.classes_).index(1)]
        iscore = personal.classifier.predict_proba(x[held])[:, list(personal.classifier.classes_).index(1)]
        c = read(args.run / "models" / fold["fold_id"] / "calibration.json")
        assert set(map(tuple, c["adaptation_records"])) == set(map(tuple, fold["adaptation_records"]))
        scores = population.classifier.predict_proba(x[adapt])[:, list(population.classifier.classes_).index(1)]
        y = meta.loc[adapt, "truth_fog_label"].to_numpy(int)
        candidates = []
        grid = config["calibration_grid"]
        for threshold in np.linspace(grid["start"], grid["stop"], grid["count"]):
            prediction = scores >= threshold
            tp, fp = int(((y == 1) & prediction).sum()), int(((y == 0) & prediction).sum())
            fn, tn = int(((y == 1) & ~prediction).sum()), int(((y == 0) & ~prediction).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0
            balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2
            candidates.append((f1, balanced, float(threshold)))
        best = max(candidates)
        assert np.isclose(best[2], c["selected"]["threshold"])
        for strategy, score, threshold in [
            ("population", pscore, config["fixed_threshold"]),
            ("personal", iscore, config["fixed_threshold"]),
            ("population_calibrated", pscore, best[2]),
        ]:
            frame = frames[strategy].loc[frames[strategy].fold_id.eq(fold["fold_id"])]
            assert len(frame) == int(held.sum())
            pd.testing.assert_frame_equal(frame[KEYS].reset_index(drop=True), meta.loc[held, KEYS].reset_index(drop=True))
            np.testing.assert_allclose(frame.fog_probability, score, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(frame.is_fog, score >= threshold)
            np.testing.assert_array_equal(frame.truth_fog_label, meta.loc[held, "truth_fog_label"])
            replayed += len(frame)
    for strategy, frame in frames.items():
        assert not frame.duplicated(KEYS + ["window_start_s", "window_end_s"]).any()
        independent = counts(frame)
        for key, value in independent.items():
            assert np.isclose(value, result["pooled"][strategy][key]), (strategy, key)
        person_f1 = []
        for person, group in frame.groupby("subject_alias"):
            actual = counts(group)
            reported = next(r for r in result["per_person"] if r["subject"] == person and r["strategy"] == strategy)
            for key, value in actual.items():
                assert (value is None and reported[key] is None) or np.isclose(value, reported[key])
            if actual["fog_f1"] is not None:
                person_f1.append(actual["fog_f1"])
        assert np.isclose(np.mean(person_f1), result["participant_macro"][strategy]["mean_participant_fog_f1"])
        runtime = pd.read_csv(args.run / f"{strategy}_runtime.csv")
        assert not any(c.startswith("truth_") for c in runtime)
    original = {}
    for folder in [root / "docs/a_group", root / "docs/a_group_showcase"]:
        for name, expected in read(folder / "MANIFEST.json")["files"].items():
            assert digest(folder / name) == expected
            original[str(folder.name) + "/" + name] = expected
    report = {"schema": "fogguard.personalized_fog.verification.v1", "status": "PASS",
              "paired_folds": result["eligible_folds"], "saved_model_predictions_replayed": replayed,
              "model_target_exclusion": "PASS", "adaptation_evaluation_block_separation": "PASS",
              "adaptation_only_threshold_search": "PASS", "independent_metric_recount": "PASS",
              "original_TEST_and_showcase_hashes_checked": len(original),
              "source_sha256": digest(Path(__file__)), "run_manifest_sha256": digest(args.run / "MANIFEST.json")}
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
