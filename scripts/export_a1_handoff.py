"""Package retained bilateral RF and synthetic interface examples without patient data."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from evaluate_a1_fixed_split import check_lock

from fogguard.modeling.a1_bilateral import DUAL, BilateralFoG


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("reports/a1_bilateral_v1_20261029"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    check_lock(args.run_dir, json.loads((args.run_dir / "MODEL_LOCK.json").read_text("utf-8")))
    source = args.run_dir / "bilateral_rf/model.joblib"
    model = BilateralFoG.load(source)
    if model.variant != "bilateral_rf" or len(model.feature_names) != 66 or not np.isclose(model.threshold, 0.23):
        raise ValueError("Unexpected retained model")
    folder = args.output / "a1-a2-handoff"
    folder.mkdir(parents=True)
    shutil.copyfile(source, folder / "model.joblib")
    t = np.arange(300) / 60
    signal = np.column_stack([40 * np.sin(2 * np.pi * (0.8 + i / 8) * t + i / 3) for i in range(6)])
    frame = pd.DataFrame(signal, columns=DUAL)
    frame.insert(0, "timestamp", t)
    frame.loc[150:179, list(DUAL[3:])] = np.nan
    frame.to_csv(folder / "synthetic_input.csv", index=False)
    runtime = model.predict_frame(frame)
    runtime["prediction_source"] = "synthetic_interface_smoke_not_performance"
    if any(c.startswith("truth_") for c in runtime) or len(runtime) != 3:
        raise ValueError("Synthetic interface or missing-input schedule changed")
    runtime.to_csv(folder / "synthetic_expected_runtime.csv", index=False)
    requirements = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    (folder / "tested-requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in requirements.items()) + "\n", "utf-8")
    readme = """# A1 to A2 model attachment
Use with matching FoGGuard source. Python 3.12. Install requirements.txt and pip install -e .
Research model: bilateral_rf, 66 features, TRAIN-only fit, cutoff 0.23. System default unchanged.
See docs/A1_A2_START_HERE.md for evidence, limits and alignment rules.

After extracting this folder under artifacts/, run from repository root:
python -X utf8 scripts/predict_a1_bilateral.py --model artifacts/a1-a2-handoff/model.joblib --input artifacts/a1-a2-handoff/synthetic_input.csv --output reports/a1-a2-smoke.csv

Example is SYNTHETIC, for interface testing only, with a right-ankle gap.
Decision available at window end. Hold at most 0.5s, then UNKNOWN.
Input: timestamp seconds, bilateral gyro xyz, 60Hz, deg/s, training axis convention.
No direction/intent/step-success output. No raw patient data or evaluation truth included.
"""
    (folder / "README.md").write_text(readme, "utf-8")
    manifest = {"version": "a1-a2-20261029", "model_variant": model.variant, "threshold": model.threshold,
        "features": 66, "channels": list(DUAL), "rate_hz": 60, "gyro_units": "deg/s", "window_samples": 120,
        "step_samples": 30, "max_hold_s": 0.5, "fit_people": list(model.fit_participants),
        "model_sha256": sha(source), "default_replaced": False, "example_source": "synthetic_interface_smoke_not_performance",
        "raw_patient_data_included": False, "files": {p.name: sha(p) for p in folder.iterdir() if p.is_file()}}
    (folder / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    target = args.output / "FoGGuard-A1-A2-handoff.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            archive.write(path, f"a1-a2-handoff/{path.name}")
    with zipfile.ZipFile(target) as archive:
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(f"a1-a2-handoff/{name}")).hexdigest() != digest:
                raise ValueError("Bundle hash differs")
    print(json.dumps({"archive": str(target), "zip_sha256": sha(target), "model_sha256": sha(source), "synthetic_windows": len(runtime)}, indent=2))


if __name__ == "__main__":
    main()
