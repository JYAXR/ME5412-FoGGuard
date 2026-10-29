"""Reproduce only the three selected methods on fixed TRAIN/VALIDATION roles; no TEST."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from fogguard.evaluation.selected_fog import annotate
from fogguard.modeling.a1_fog import prediction_frame
from fogguard.selected_fog import METHODS, matrix_for, sha, windows_for, wrap_classifier, write_json


def choose(labels, scores, config):
    grid = config["threshold_grid"]
    candidates = []
    for threshold in np.linspace(grid["start"], grid["stop"], grid["count"]):
        decision = scores >= threshold
        candidates.append({"threshold": float(threshold), "f1": f1_score(labels, decision, zero_division=0),
                           "balanced_accuracy": balanced_accuracy_score(labels, decision)})
    return max(candidates, key=lambda r: (r["f1"], r["balanced_accuracy"], r["threshold"])), candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/fog_selected.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new training directory")
    config = json.loads(args.config.read_text("utf-8"))
    if sha(args.raw) != config["raw_sha256"]:
        raise ValueError("Different raw dataset")
    raw = pd.read_csv(args.raw)
    parts = {role: raw.loc[raw.subjectID.isin(config["split"][role])].copy() for role in ("train", "validation")}
    del raw
    args.output.mkdir(parents=True)
    for method in METHODS if args.method == "all" else (args.method,):
        train = windows_for(parts["train"], method)
        x, names = matrix_for(train, method)
        y = np.array([w.fog_label for w in train])
        people = np.array([int(w.subject_id) for w in train])
        folder = args.output / method
        folder.mkdir()
        if method == "bilateral_rf":
            scores, seen = np.full(len(y), np.nan), np.zeros(len(y), int)
            for plan in config["bilateral_folds"]:
                fit, held = [np.flatnonzero(np.isin(people, plan[k])) for k in ("fit_people", "held_out_people")]
                if set(plan["fit_people"]) & set(plan["held_out_people"]):
                    raise ValueError("Person overlap")
                forest = RandomForestClassifier(**config["rf"], random_state=config["random_state"], n_jobs=-1).fit(x[fit], y[fit])
                scores[held], seen[held] = forest.predict_proba(x[held])[:, 1], seen[held] + 1
                joblib.dump(forest, folder / f"fold_{plan['fold']}.joblib")
            if not np.all(seen == 1):
                raise ValueError("Incomplete TRAIN OOF")
            selected, candidates = choose(y, scores, config)
            annotate(train, prediction_frame(train, scores, selected["threshold"], method), "train_oof_calibration").to_csv(folder / "train_oof.csv", index=False)
            classifier = RandomForestClassifier(**config["rf"], random_state=config["random_state"], n_jobs=-1).fit(x, y)
            threshold_people = config["split"]["train"]
        else:
            val = windows_for(parts["validation"], method)
            vx, _ = matrix_for(val, method)
            vy = np.array([w.fog_label for w in val])
            if method == "ankle_cnn":
                from fogguard.modeling.a1_author_cnn import AuthorCNNClassifier

                classifier = AuthorCNNClassifier(config["cnn"], config["random_state"]).fit(x, y, (vx, vy))
                pd.DataFrame(classifier.history_).to_csv(folder / "training_history.csv", index=False)
            else:
                classifier = RandomForestClassifier(**config["rf"], random_state=config["random_state"], n_jobs=-1).fit(x, y)
            selected, candidates = choose(vy, classifier.predict_proba(vx)[:, 1], config)
            threshold_people = config["split"]["validation"]
        model = wrap_classifier(classifier, method, train[0].channel_names, names, selected["threshold"], sorted(map(int, set(people))), threshold_people)
        model.save(folder / "model.joblib")
        write_json(folder / "fit_metadata.json", {"method": method, "fit_people": list(model.fit_participants),
            "threshold_people": list(model.threshold_participants), "fit_windows": len(y), "features": len(names),
            "best_epoch": getattr(classifier, "best_epoch_", None), "test_people_used": [],
            "threshold_policy": config["threshold_policies"][method]})
        write_json(folder / "threshold_selection.json", {"selected": selected, "candidates": candidates,
            "policy": config["threshold_policies"][method], "people": threshold_people})
        print(f"{method}: saved, threshold {model.threshold:.2f}", flush=True)
    write_json(args.output / "MANIFEST.json", {"methods": list(METHODS if args.method == "all" else (args.method,)),
        "scope": "Development reproduction, differing declared threshold policies retained; no TEST",
        "test_windows_constructed": 0, "raw_sha256": sha(args.raw), "config_sha256": sha(args.config),
        "files": {p.relative_to(args.output).as_posix(): sha(p) for p in args.output.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    main()
