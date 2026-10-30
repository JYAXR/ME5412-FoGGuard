"""Preregister and run blocked personalization comparisons inside TRAIN14 only."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.evaluation.heldout import attach_fog_truth, binary_report
from fogguard.evaluation.intervals import EventSupport, merge
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_fog import a1_matrix
from fogguard.modeling.personalized_fog import (
    KEYS,
    adaptation_status,
    choose_calibration,
    fit_model,
    plan_blocks,
    record_mask,
)
from fogguard.selected_fog import sha

RAW_KEYS = ["subjectID", "sessionID", "taskID"]
RUNTIME_KEYS = KEYS + ["window_start_s", "window_end_s", "available_at_s"]


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def write(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                           + "\n").encode("utf-8"))


def source_hashes(root):
    names = ["config/personalized_fog.json", "config/fog_selected.json",
             "scripts/run_personalized_fog.py", "src/fogguard/modeling/personalized_fog.py",
             "src/fogguard/modeling/a1_bilateral.py", "src/fogguard/modeling/a1_fog.py",
             "src/fogguard/modeling/fogstar_baseline.py", "src/fogguard/data/fogstar.py",
             "src/fogguard/evaluation/heldout.py", "src/fogguard/evaluation/intervals.py",
             "src/fogguard/evaluation/selected_fog.py"]
    return {name: sha(root / name) for name in names}


def load_train(raw, split):
    return pd.concat([c.loc[c.subjectID.isin(split["train"])] for c in pd.read_csv(
        raw, usecols=["timestamp", *RAW_KEYS, *DUAL, "fog"], chunksize=50000)], ignore_index=True)


def metric_report(frame):
    report = binary_report(frame)
    negative = report["TN"] + report["FP"]
    report["false_positive_rate"] = report["FP"] / negative if negative else None
    if not report["positive_windows"]:
        report["fog_f1"] = report["recall"] = None
        report["positive_class_evaluation"] = "NOT_EVALUABLE: no positive windows"
    return report


def run(args, root, config, base):
    out = args.output_dir
    lock = read(out / "PLAN_LOCK.json")
    if lock["source_sha256"] != source_hashes(root) or lock["raw_sha256"] != sha(args.raw):
        raise ValueError("Code/config/data changed after planning")
    for name, digest in lock["inputs"].items():
        if sha(out / name) != digest:
            raise ValueError(f"Planned input changed: {name}")
    started = out / "RUN_STARTED.json"
    with started.open("x", encoding="utf-8") as handle:
        json.dump({"plan_lock_sha256": sha(out / "PLAN_LOCK.json"),
                   "started_at_utc": datetime.now(timezone.utc).isoformat()}, handle)
    cache = np.load(out / "features.npz", allow_pickle=False)
    x, names = cache["values"], tuple(cache["names"].tolist())
    meta = pd.read_csv(out / "window_metadata.csv", dtype={k: str for k in KEYS})
    folds = read(out / "folds.json")
    aliases = {str(p): f"S{i + 1:02d}" for i, p in enumerate(base["split"]["train"])}
    (out / "models").mkdir()
    write(out / "private_subject_aliases.json", aliases)
    rows = {s: [] for s in config["strategies"]}
    population_all, fold_reports = [], []
    all_people = set(meta.subject_id)
    for target in map(str, base["split"]["train"]):
        target_mask = meta.subject_id.eq(target).to_numpy()
        other = ~target_mask
        model = fit_model(x[other], meta.loc[other, "truth_fog_label"], names, config,
                          strategy="population", fit_people=sorted(all_people - {target}))
        if target in model.fit_participants:
            raise AssertionError("Population model contains target participant")
        population_path = out / "models" / f"population_p{target}.joblib"
        model.save(population_path)
        scores = model.predict_features(x[target_mask])
        full_scores = np.full(len(meta), np.nan)
        full_scores[target_mask] = scores
        population = meta.loc[target_mask].copy()
        population["fog_probability"] = scores
        population["is_fog"] = scores >= config["fixed_threshold"]
        population_all.append(population)
        selected_folds = [f for f in folds if f["subject_id"] == target]
        print(f"{aliases[target]}: target excluded from population fit; "
              f"{sum(f['eligible'] for f in selected_folds)}/{len(selected_folds)} eligible blocks", flush=True)
        for fold in selected_folds:
            if not fold["eligible"]:
                fold_reports.append(fold)
                continue
            adapt = record_mask(meta, fold["adaptation_records"])
            held = record_mask(meta, fold["evaluation_records"])
            if (np.any(adapt & held) or not np.all(target_mask[adapt | held])
                    or not np.isfinite(full_scores[adapt | held]).all()):
                raise AssertionError("Fold leakage or person mismatch")
            personal = fit_model(x[adapt], meta.loc[adapt, "truth_fog_label"], names, config,
                                 strategy="personal", fit_people=[target],
                                 adaptation_records=fold["adaptation_records"])
            folder = out / "models" / fold["fold_id"]
            folder.mkdir()
            personal.save(folder / "personal.joblib")
            best, search = choose_calibration(meta.loc[adapt, "truth_fog_label"], full_scores[adapt],
                                              config["calibration_grid"])
            write(folder / "calibration.json", {
                "base_model": f"../{population_path.name}", "selected": best,
                "adaptation_records": fold["adaptation_records"], "candidates": search,
                "threshold_fit_scope": "Target adaptation blocks only; no evaluation labels",
            })
            choices = {
                "population": (full_scores[held], config["fixed_threshold"]),
                "personal": (personal.predict_features(x[held]), config["fixed_threshold"]),
                "population_calibrated": (full_scores[held], best["threshold"]),
            }
            for strategy, (score, threshold) in choices.items():
                evaluated = meta.loc[held].copy()
                evaluated["fog_probability"], evaluated["fog_threshold"] = score, threshold
                evaluated["is_fog"] = score >= threshold
                evaluated["fold_id"], evaluated["split_kind"] = fold["fold_id"], fold["split_kind"]
                evaluated["subject_alias"], evaluated["strategy"] = aliases[target], strategy
                evaluated["model_version"] = f"personal_study_v1/{strategy}"
                evaluated["prediction_source"] = "exploratory_train_pool_blocked_cv"
                rows[strategy].append(evaluated)
            fold_reports.append({**fold, "calibrated_threshold": best["threshold"],
                                 "personal_fit_windows": int(adapt.sum())})
    pd.concat(population_all, ignore_index=True).to_csv(out / "population_all_evaluation.csv", index=False)
    write(out / "fold_results.json", fold_reports)
    if not rows["personal"]:
        raise ValueError("No paired folds support the three-way study; inspect the feasibility plan")
    frames = {s: pd.concat(parts, ignore_index=True) for s, parts in rows.items()}
    first = frames["population"]
    keys = KEYS + ["window_start_s", "window_end_s"]
    if first.duplicated(keys).any():
        raise AssertionError("A held-out window was evaluated twice")
    for strategy, evaluated in frames.items():
        pd.testing.assert_frame_equal(first[keys + ["truth_fog_label"]],
                                      evaluated[keys + ["truth_fog_label"]])
        evaluated.to_csv(out / f"{strategy}_evaluation.csv", index=False)
        evaluated.drop(columns=[c for c in evaluated if c.startswith("truth_")]).to_csv(
            out / f"{strategy}_runtime.csv", index=False)
    raw = load_train(args.raw, base["split"])
    eval_identities = first[KEYS].drop_duplicates().astype(str)
    raw_keys = raw[RAW_KEYS].rename(columns=dict(zip(RAW_KEYS, KEYS, strict=True)))
    raw = raw.loc[record_mask(raw_keys, eval_identities.values.tolist())]
    allowed = sorted(raw.subjectID.unique().tolist())
    support = EventSupport.build(raw, first, allowed=allowed,
                                 forbidden=base["split"]["validation"] + base["split"]["test"])
    observed = sum(sum(b - a for a, b in merge(zip(g.timestamp, g.timestamp + 1 / 60, strict=True)))
                   for _, g in raw.groupby(RAW_KEYS, sort=False))
    per_person, per_split, overall = [], [], {}
    for strategy, evaluated in frames.items():
        overall[strategy] = {**metric_report(evaluated), **support.metrics(evaluated.is_fog),
                             "observed_s": observed,
                             "false_alarm_s_per_observed_minute": 60 * support.metrics(evaluated.is_fog)["false_alarm_s"] / observed}
        for alias, group in evaluated.groupby("subject_alias", sort=True):
            per_person.append({"subject": alias, "strategy": strategy,
                               "split_kind": group.split_kind.iloc[0], **metric_report(group)})
        for kind, group in evaluated.groupby("split_kind", sort=True):
            per_split.append({"split_kind": kind, "strategy": strategy,
                              "participants": group.subject_alias.nunique(), **metric_report(group)})
    per_person_frame = pd.DataFrame(per_person)
    macro = {}
    for strategy in config["strategies"]:
        group = per_person_frame.loc[per_person_frame.strategy.eq(strategy)]
        positive = group.loc[group.positive_windows.gt(0)]
        macro[strategy] = {"positive_participants": len(positive),
                           "mean_participant_fog_f1": float(positive.fog_f1.mean()),
                           "mean_participant_recall": float(positive.recall.mean()),
                           "mean_participant_false_positive_rate": float(group.false_positive_rate.mean())}
    all_population = pd.concat(population_all, ignore_index=True)
    controls = []
    for person, group in all_population.groupby("subject_id", sort=True):
        if not group.truth_fog_label.any():
            controls.append({"subject": aliases[person], **metric_report(group)})
    status_counts = pd.Series([f["reason"] for f in folds]).value_counts().to_dict()
    summary = {
        "schema": "fogguard.personalized_fog.results.v1", "scope": config["interpretation"],
        "original_TEST_scored": False, "original_validation_scored": False,
        "cohort_participants": len(aliases), "cohort_records": len(read(out / "records.json")),
        "cohort_supported_windows": len(meta), "paired_participants": first.subject_alias.nunique(),
        "paired_records": len(first[KEYS].drop_duplicates()), "paired_windows": len(first),
        "paired_positive_windows": int(first.truth_fog_label.sum()), "fold_status": status_counts,
        "eligible_folds": int(sum(f["eligible"] for f in folds)), "planned_folds": len(folds),
        "pooled": overall, "participant_macro": macro, "per_person": per_person,
        "per_split_kind": per_split, "negative_only_population_controls": controls,
        "limitations": ["Eligibility is based only on adaptation support; excluded folds are reported",
                        "Session IDs are grouping keys, not verified dates; record holdout is within-session",
                        "Overlapping windows are not independent participants",
                        "Personal and population fits use different amounts of data; this does not prove the cause of errors",
                        "Population calibration uses labeled target data and is not zero-shot generalization",
                        "Existing TRAIN pool and prior design knowledge: not a fresh clinical or blind test"],
    }
    write(out / "results.json", summary)
    pd.DataFrame([{ "strategy": strategy, **values} for strategy, values in overall.items()]).to_csv(
        out / "comparison.csv", index=False)
    per_person_frame.to_csv(out / "per_person.csv", index=False)
    if lock["source_sha256"] != source_hashes(root):
        raise ValueError("Source changed during training")
    write(out / "MANIFEST.json", {"schema": "fogguard.personalized_fog.run.v1",
        "plan_lock_sha256": sha(out / "PLAN_LOCK.json"), "source_sha256": lock["source_sha256"],
        "files": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()},
        "original_models_replaced": False, "strategy_count": len(config["strategies"])})
    print(pd.DataFrame(overall).T[["precision", "recall", "fog_f1", "false_positive_rate"]].to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "run"])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base, config = read(root / "config/fog_selected.json"), read(root / "config/personalized_fog.json")
    if sha(args.raw) != base["raw_sha256"]:
        raise ValueError("Raw source differs from fixed dataset")
    if config["cohort_role"] != "train" or not config["original_validation_and_test_excluded"]:
        raise ValueError("Only the original TRAIN pool is authorized for this exploratory study")
    if args.mode == "run":
        run(args, root, config, base)
        return
    if args.output_dir.exists():
        parser.error("Use a fresh plan directory")
    raw = load_train(args.raw, base["split"])
    sensors = raw.drop(columns="fog")
    windows = anchored_gyro_windows(sensors, require_right=True)
    x, names = a1_matrix(windows, "baseline_2s_v1")
    meta = pd.DataFrame([{**dict(zip(KEYS, (w.subject_id, w.session_id, w.task_id), strict=True)),
                          "window_start_s": w.start_s, "window_end_s": w.end_s, "available_at_s": w.end_s}
                         for w in windows])
    meta = attach_fog_truth(meta, anchored_gyro_windows(raw, require_right=True))
    records = raw[RAW_KEYS].rename(columns=dict(zip(RAW_KEYS, KEYS, strict=True))).astype(str).drop_duplicates()
    folds = [{**fold, **adaptation_status(meta, fold, config)} for fold in plan_blocks(records)]
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "features.npz", values=x, names=np.array(names))
    meta.to_csv(args.output_dir / "window_metadata.csv", index=False)
    write(args.output_dir / "records.json", records.values.tolist())
    write(args.output_dir / "folds.json", folds)
    write(args.output_dir / "PLAN_LOCK.json", {"schema": "fogguard.personalized_fog.plan.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "raw_sha256": sha(args.raw),
        "source_sha256": source_hashes(root), "config": config,
        "original_cohort_role": "train", "original_validation_and_test_used": False,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "inputs": {name: sha(args.output_dir / name) for name in
                   ("features.npz", "window_metadata.csv", "records.json", "folds.json")},
        "fitted_models_at_plan_creation": 0})
    print(json.dumps({"supported_windows": len(meta), "records": len(records),
                      "planned_folds": len(folds), "eligible_folds": sum(f["eligible"] for f in folds),
                      "status_counts": pd.Series([f["reason"] for f in folds]).value_counts().to_dict()}), flush=True)


if __name__ == "__main__":
    main()
