"""Package only the three selected model files and synthetic interface examples."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.data.fogstar import sensor_columns
from fogguard.selected_fog import METHODS, check_lock, load_selected, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    check_lock(args.models, json.loads((args.models / "MANIFEST.json").read_text("utf-8")))
    folder = args.output / "fog-selected"
    folder.mkdir(parents=True)
    t = np.arange(420) / 60
    channels = (*sensor_columns(("ankleL",)), *(f"ankleR_gyro_{a}" for a in "xyz"))
    data = {c: (0.2 if "_acc_" in c else 40) * np.sin(2 * np.pi * (0.8 + i / 8) * t + i / 3) for i, c in enumerate(channels)}
    frame = pd.DataFrame(data).assign(timestamp=t)
    frame.loc[150:179, [f"ankleR_gyro_{a}" for a in "xyz"]] = np.nan
    frame.to_csv(folder / "synthetic_input.csv", index=False)
    metadata = {}
    for method in METHODS:
        dest = folder / method
        dest.mkdir()
        shutil.copyfile(args.models / method / "model.joblib", dest / "model.joblib")
        model = load_selected(dest / "model.joblib", method)
        result = model.predict_frame(frame)
        result["prediction_source"] = "synthetic_interface_smoke_not_performance"
        result.to_csv(dest / "synthetic_expected_runtime.csv", index=False)
        metadata[method] = {"channels": list(model.channel_names), "threshold": model.threshold,
            "features": len(model.feature_names), "fit_people": list(model.fit_participants),
            "threshold_people": list(model.threshold_participants), "model_sha256": sha(dest / "model.joblib"),
            "synthetic_rows": len(result)}
    shutil.copyfile("docs/A1_A2_START_HERE.md", folder / "START_HERE.md")
    shutil.copyfile("docs/A1_A2_START_HERE.md", folder / "A1_A2_START_HERE.md")
    (folder / "a1").mkdir()
    shutil.copyfile("docs/a1/SELECTED_METHODS.md", folder / "a1/SELECTED_METHODS.md")
    shutil.copyfile("config/fog_selected.json", folder / "training_config.json")
    import joblib
    import scipy
    import sklearn
    import torch

    write_json(folder / "TESTED_VERSIONS.json", {"python": "3.12", "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__, "torch": torch.__version__})
    write_json(folder / "MANIFEST.json", {"schema": "fog-selected.v1", "methods": metadata,
        "sample_rate_hz": 60, "window_samples": 120, "step_samples": 30, "max_hold_s": 0.5,
        "raw_patient_data_included": False, "example_source": "synthetic_interface_smoke_not_performance",
        "files": {str(p.relative_to(folder)): sha(p) for p in folder.rglob("*") if p.is_file()}})
    archive = args.output / "FoGGuard-Selected-3-Models.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, "fog-selected/" + p.relative_to(folder).as_posix())
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("fog-selected/MANIFEST.json"))
        import hashlib

        for relative, digest in manifest["files"].items():
            if hashlib.sha256(z.read("fog-selected/" + relative.replace("\\", "/"))).hexdigest() != digest:
                raise ValueError("Bundle hash differs")
    print(json.dumps({"archive": str(archive), "sha256": sha(archive), "methods": list(metadata)}, indent=2))


if __name__ == "__main__":
    main()
