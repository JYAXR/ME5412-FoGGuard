"""Prepare every manifest source sensor stream, without decoding target labels."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from run_a1_experiment import sha, write_json

from fogguard.data.stanford_external import (
    FILTER,
    SOURCE_GYRO,
    read_workbook,
    resample_gyro,
    source_identity,
)
from fogguard.modeling.a1_bilateral import DUAL, LEFT, anchored_gyro_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/a1_stanford_external_v1.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/stanford_external_v1_20261029"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root, run = Path(__file__).resolve().parents[1], args.run_dir
    if run.exists():
        parser.error("Use a new input run")
    config = json.loads(args.config.read_text("utf-8"))
    manifest = json.loads(Path(config["source_manifest"]).read_text("utf-8"))
    if sha(config["model"]) != config["model_sha256"]:
        raise ValueError("Frozen model changed")
    visits = {}
    for entry in manifest["files"]:
        person, visit = source_identity(Path(entry["path"]).name)
        visits.setdefault(person, set()).add(visit)
    run.mkdir(parents=True)
    (run / "inputs").mkdir()
    write_json(run / "config.json", config)
    write_json(run / "source_manifest.json", manifest)
    rows = []
    for i, entry in enumerate(manifest["files"]):
        name = Path(entry["path"]).name
        path = args.source_dir / name
        blob = path.read_bytes()
        if len(blob) != entry["size"] or hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest() != entry["git_blob_sha1"]:
            raise ValueError("Source workbook hash differs")
        data = read_workbook(path, labels=False)
        person, visit = source_identity(name)
        if not np.all(data["subject_ID"] == person):
            raise ValueError("Workbook subject differs from filename")
        gyro = np.column_stack([data[c] for c in SOURCE_GYRO])
        frame, time, blocks = resample_gyro(data["time"], gyro)
        ids = [1000 + person, sorted(visits[person]).index(visit) + 1, i + 1]
        for column, value in zip(("subjectID", "sessionID", "taskID"), ids, strict=True):
            frame[column] = value
        windows = anchored_gyro_windows(frame, require_right=True) if len(frame) else []
        np.savez_compressed(run / "inputs" / f"record_{i + 1:03}.npz", native_time=time,
            logical_time=frame.timestamp.to_numpy(float), gyro=frame[list(DUAL)].to_numpy(float))
        rows.append({"record_id": i + 1, "filename": name, "source_person": person, "source_visit": visit,
            "subject_id": ids[0], "session_id": ids[1], "task_id": ids[2], "source_sha256": sha(path),
            "native_samples": len(time), "clock_blocks": len(blocks), "resampled_rows": len(frame),
            "max_clock_reconstruction_s": float(np.max(abs(time - data["time"]))),
            "finite_left_rows": int(np.isfinite(frame[list(LEFT)].to_numpy(float)).all(axis=1).sum()),
            "finite_bilateral_rows": int(np.isfinite(frame[list(DUAL)].to_numpy(float)).all(axis=1).sum()), "windows": len(windows)})
        print(f"Sensors only {i + 1}/{len(manifest['files'])}: {name}, windows={len(windows)}", flush=True)
    pd.DataFrame(rows).to_csv(run / "input_inventory.csv", index=False)
    sources = [root / f"scripts/{name}.py" for name in ("prepare_stanford_external", "evaluate_stanford_external", "verify_stanford_external", "report_stanford_external")]
    sources += sorted((root / "src/fogguard").rglob("*.py"))
    write_json(run / "protocol.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha(args.config), "model_sha256": sha(config["model"]),
        "filter_sha256": hashlib.sha256(FILTER.tobytes()).hexdigest(), "target_label_values_decoded": False,
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in sources}})
    write_json(run / "INPUT_LOCK.json", {"locked_utc": datetime.now(timezone.utc).isoformat(),
        "target_label_values_decoded": False, "model_sha256": sha(config["model"]), "source_records": len(rows),
        "files": {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()}})
    print(f"All {len(rows)} sensor records locked; external labels may now be decoded in evaluation", flush=True)


if __name__ == "__main__":
    main()
