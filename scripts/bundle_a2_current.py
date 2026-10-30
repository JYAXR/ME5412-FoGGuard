"""Publish the explicitly approved A2 current model with a synthetic handoff."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.turning.learned import DirectionModel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("reports/a2_final_20261030/current/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("models/a2/current"))
    args = parser.parse_args()
    target = args.output / "model.joblib"
    if target.exists():
        parser.error("Published target already exists; use a fresh bundle directory")
    model = DirectionModel.load(args.source)
    if model.horizon_s != 0 or model.classifier.n_features_in_ != 12:
        raise ValueError("Expected the selected current-direction model")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, target)
    assert sha(target) == sha(args.source)
    shutil.copyfile("config/a2_turning.json", args.output / "training_config.json")
    versions = {name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]}
    (args.output / "requirements.txt").write_text(
        "# Tested inference dependencies; Python 3.12.13\n"
        + "".join(f"{name}=={version}\n" for name, version in versions.items()), "utf-8")
    t = np.arange(240) / 60
    example = pd.DataFrame({"timestamp": t, "subjectID": 0, "sessionID": 0, "taskID": 0})
    for channel in model.channels:
        example[channel] = 0.0
    for position in ["ankleL", "back"]:
        example[f"{position}_acc_x"] = -1.0
        example[f"{position}_acc_y"] = 0.05 * np.sin(2*np.pi*1.5*t)
        example[f"{position}_gyro_z"] = 5 * np.sin(2*np.pi*1.5*t)
    example_path = args.output / "synthetic_input.csv"
    example.to_csv(example_path, index=False)
    # Verify the bytes receivers will read, not an in-memory unrounded variant.
    supplied = pd.read_csv(example_path)
    output = DirectionModel.load(target).predict_frame(supplied)
    assert len(output) == 5
    np.testing.assert_allclose(output.available_at_s, 119/60 + np.arange(5)*0.5)
    assert not set(output.columns) & {"activity", "fog", "target"}
    output.to_csv(args.output / "synthetic_expected.csv", index=False)
    files = [target, args.output / "training_config.json", args.output / "requirements.txt",
             example_path, args.output / "synthetic_expected.csv"]
    metadata = {
        "schema": "fogguard.a2.current.bundle.v1", "task": "current_direction",
        "model_file": "model.joblib", "model_sha256": sha(target), "model_bytes": target.stat().st_size,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_files_git_blob_sha256": {
            p: hashlib.sha256(subprocess.check_output(["git", "show", "HEAD:"+p])).hexdigest()
            for p in ["src/fogguard/turning/learned.py", "src/fogguard/data/fogstar.py"]},
        "bundle_script_sha256": sha(Path(__file__)),
        "versions": {"python": platform.python_version(), **versions},
        "features": 12, "trees": model.classifier.n_estimators,
        "feature_indices": list(model.feature_indices), "rejection_threshold": model.rejection_threshold,
        "channels": list(model.channels), "classes": model.classifier.classes_.tolist(),
        "sample_rate_hz": 60, "window_samples": 120, "step_samples": 30,
        "synthetic_sample_only": True, "synthetic_smoke_check": "PASS",
        "evaluation_scope": "TRAIN14 / development validation4; TEST not scored",
        "known_direction_accuracy": 0.6699115044247788,
        "known_direction_macro_f1": 0.7197309824304418,
        "publication_exception": "User explicitly requested this current-direction model in Git",
        "files_sha256": {p.name: sha(p) for p in files},
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    print(json.dumps({"path": target.as_posix(), "sha256": sha(target),
                      "bytes": target.stat().st_size, "smoke_rows": len(output)}, indent=2))


if __name__ == "__main__":
    main()
