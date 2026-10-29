"""Three isolated TRAIN-only objectives with identical CNN-GRU / auxiliary heads."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json

from fogguard.evaluation.a1_author import annotate
from fogguard.evaluation.a1_fixed_split import choose_threshold, role_frame
from fogguard.evaluation.a1_temporal import assert_history_isolation, person_indices
from fogguard.modeling.a1_author_cnn import GYRO_CHANNELS, gyro_windows
from fogguard.modeling.a1_context_learning import (
    ARMS,
    ContextClassifier,
    ContextFoG,
    activity_targets,
)
from fogguard.modeling.a1_fixed_split import prediction_frame
from fogguard.modeling.a1_fog import A1FogModel
from fogguard.modeling.a1_temporal_cnn import sequence_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_context_learning_v1.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    if run.exists() and any(run.iterdir()):
        parser.error("Preserve previous runs; use a fresh directory")
    config = json.loads(args.config.read_text("utf-8"))
    if config["models"] != list(ARMS) or config["arms"] != {
        "control": {"person_balance": False, "activity_weight": 0.0},
        "person_balanced": {"person_balance": True, "activity_weight": 0.0},
        "activity_aux": {"person_balance": False, "activity_weight": 0.2}}:
        raise ValueError("Unexpected isolated arm configuration")
    reference = Path(config["reference_run"])
    check_lock(reference, json.loads((reference / "MODEL_LOCK.json").read_text("utf-8")))
    previous = json.loads((reference / "protocol.json").read_text("utf-8"))
    split = json.loads(Path(config["split_config"]).read_text("utf-8"))["split"]
    if split != json.loads((reference / "split.json").read_text("utf-8")) or sha(args.raw) != previous["raw_sha256"]:
        raise ValueError("Data or participant roles changed")
    raw = pd.read_csv(args.raw)
    windows = gyro_windows(role_frame(raw, split, "train"))
    del raw
    x, lengths, links = sequence_batch(windows)
    groups = np.array([int(w.subject_id) for w in windows])
    y, activities = np.array([w.fog_label for w in windows]), activity_targets(windows)
    if set(groups) != set(split["train"]):
        raise ValueError("TRAIN people missing")
    plans = json.loads((reference / "fold_plan.json").read_text("utf-8"))
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", config)
    write_json(run / "split.json", split)
    write_json(run / "fold_plan.json", plans)
    np.savez_compressed(run / "training_metadata.npz", links=links, lengths=lengths, people=groups, activities=activities)
    source = [root / f"scripts/{n}.py" for n in ("train_a1_context_learning", "evaluate_a1_context_learning", "verify_a1_context_learning",
              "report_a1_author_cnn", "verify_a1_author_cnn", "verify_a1_temporal_cnn", "run_a1_event")]
    source += [root / "recognition/diagnostics.py", *sorted((root / "src/fogguard").rglob("*.py"))]
    write_json(run / "protocol.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha(args.config), "split_sha256": sha(config["split_config"]), "raw_sha256": sha(args.raw),
        "reference_model_lock_sha256": sha(reference / "MODEL_LOCK.json"), "fold_plan_sha256": sha(reference / "fold_plan.json"),
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in source}, "data_roles_constructed": ["train"],
        "validation_windows_before_lock": 0, "test_windows_constructed": 0, "fit_windows": len(windows)})
    print(f"TRAIN only {len(windows)} windows, three objective arms; no VALIDATION/TEST extraction", flush=True)
    for name in config["models"]:
        directory = run / name
        directory.mkdir()
        scores, seen, epochs = np.full(len(windows), np.nan), np.zeros(len(windows), int), []
        for plan in plans:
            fold = plan["fold"]
            dest = directory / f"fold_{fold}"
            dest.mkdir()
            indices = {key: person_indices(groups, plan[key]) for key in
                       ("inner_fit_people", "checkpoint_people", "fit_people", "held_out_people")}
            for key, index in indices.items():
                assert_history_isolation(links, groups, index, plan[key])
            inner, check, fit, held = [indices[k] for k in ("inner_fit_people", "checkpoint_people", "fit_people", "held_out_people")]
            seed = config["random_state"] + fold

            def classifier(arm=name, random_seed=seed):
                return ContextClassifier(arm, config["optimization"], random_seed, config["arms"][arm])

            print(f"{name} fold{fold}: INNER fit/checkpoint disjoint; checkpoint uses only FoG", flush=True)
            tuned = classifier().fit(x[inner], lengths[inner], y[inner], groups=groups[inner], activities=activities[inner],
                                     validation=(x[check], lengths[check], y[check]))
            epoch = tuned.best_epoch_
            epochs.append(epoch)
            pd.DataFrame(tuned.history_).to_csv(dest / "checkpoint_history.csv", index=False)
            pd.DataFrame(tuned.weight_audit_).to_csv(dest / "inner_person_weights.csv", index=False)
            joblib.dump(tuned, dest / "checkpoint_model.joblib")
            print(f"{name} fold{fold}: fresh outer refit {epoch} epochs", flush=True)
            model = classifier().fit(x[fit], lengths[fit], y[fit], groups=groups[fit], activities=activities[fit], fixed_epochs=epoch)
            pd.DataFrame(model.history_).to_csv(dest / "refit_history.csv", index=False)
            pd.DataFrame(model.weight_audit_).to_csv(dest / "outer_person_weights.csv", index=False)
            joblib.dump(model, dest / "oof_model.joblib")
            scores[held] = model.predict_proba(x[held], lengths[held])[:, 1]
            seen[held] += 1
            write_json(dest / "fit_audit.json", {**plan, "seed": seed, "best_epoch": epoch,
                "objective": config["arms"][name], "checkpoint_loss": "unweighted FoG BCE + kernel L2 only",
                "checkpoint_activity_targets_used": False, "checkpoint_weights_used_for_oof": False})
        if not np.all(seen == 1) or not np.isfinite(scores).all():
            raise ValueError("Incomplete TRAIN OOF")
        choice = choose_threshold(y, scores, config)
        cutoff, epoch = choice["selected"]["threshold"], int(np.median(epochs))
        out = directory / "train_oof"
        out.mkdir()
        predictions = annotate(windows, prediction_frame(windows, scores, cutoff, name), "train_oof")
        predictions["prediction_source"] = "participant_held_out_train_oof"
        predictions["model_version"] = f"a1_context_learning_v1/{name}"
        predictions["context_windows"] = lengths
        membership = {p: a["fold"] for a in plans for p in a["held_out_people"]}
        predictions["held_out_fold"] = predictions.subject_id.astype(int).map(membership)
        predictions.to_csv(out / "predictions.csv", index=False)
        predictions.drop(columns=[c for c in predictions if c.startswith("truth_")]).to_csv(out / "runtime.csv", index=False)
        write_json(directory / "threshold_selection.json", choice)
        print(f"{name}: final TRAIN14 fresh fit {epoch} epochs, TRAIN OOF threshold={cutoff:.2f}", flush=True)
        model = ContextClassifier(name, config["optimization"], config["random_state"], config["arms"][name]).fit(
            x, lengths, y, groups=groups, activities=activities, fixed_epochs=epoch)
        pd.DataFrame(model.history_).to_csv(directory / "final_history.csv", index=False)
        pd.DataFrame(model.weight_audit_).to_csv(directory / "person_weights.csv", index=False)
        detector = ContextFoG(model, name, GYRO_CHANNELS, (), cutoff, tuple(split["train"]), tuple(split["train"]))
        detector.save(directory / "model.joblib")
        restored = A1FogModel.load(directory / "model.joblib")
        pd.testing.assert_frame_equal(restored.predict_windows(windows[:20]), detector.predict_windows(windows[:20]))
        write_json(directory / "fit_metadata.json", {"fit_people": split["train"], "threshold_people": split["train"],
            "epochs": epoch, "inner_best_epochs": epochs, "parameters": model.parameter_count_, "fit_windows": len(windows),
            "objective": config["arms"][name], "activity_target_counts": model.activity_target_counts_,
            "validation_people_used": [], "test_people_used": [], "threshold_source": "TRAIN participant-held-out OOF",
            "default_replaced": False, "final_weights_source": "fresh initialization; no checkpoint/fold transfer"})
    (run / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    write_json(run / "MODEL_LOCK.json", {"locked_utc": datetime.now(timezone.utc).isoformat(), "fit_people": split["train"],
        "epoch_and_threshold_people": split["train"], "validation_windows_before_lock": 0, "test_windows_constructed": 0,
        "files": {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()}})
    print("MODEL LOCK written. Independent validation evaluation can start.", flush=True)


if __name__ == "__main__":
    main()
