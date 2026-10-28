"""One-command reproduction of the pre-work audit and algorithm evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path,
                        help="Reuse saved baseline_training.json and baseline_oof.csv; no retraining")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir or Path("reports") / datetime.now().strftime(
        "readiness_%Y%m%d_%H%M%S")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    if not args.raw.is_file():
        parser.error("Raw data missing; first run scripts/download_fogstar.py")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"),
               MPLCONFIGDIR=str(root / "artifacts" / "matplotlib-readiness-cache"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    training = output / "baseline_training.json"
    oof = output / "baseline_oof.csv"
    results = {}

    def run(name, arguments):
        print(f"Running {name} ...", flush=True)
        process = subprocess.run([sys.executable, "-X", "utf8", *arguments], cwd=root,
                                 env=env, capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", check=False)
        (logs / f"{name}.txt").write_text(process.stdout + process.stderr, encoding="utf-8")
        results[name] = {"exit_code": process.returncode}
        (output / "run_status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"{name} failed; inspect {logs / (name + '.txt')}")
        print(f"{name}: PASS", flush=True)

    if args.baseline_dir:
        for name in ("baseline_training.json", "baseline_oof.csv"):
            shutil.copy2(args.baseline_dir / name, output / name)
        results["training"] = {"status": "REUSED_SAVED_OOF", "retrained_in_this_run": False}
    else:
        run("training", ["scripts/train_fogstar_baseline.py", "--data", str(args.raw),
                         "--model-output", str(output / "baseline.joblib"),
                         "--report-output", str(training), "--oof-output", str(oof)])
    run("recognition", ["recognition/evaluate.py", "--raw", str(args.raw),
                        "--predictions", str(oof), "--training-report", str(training),
                        "--output", str(output / "recognition")])
    run("demo", ["scripts/run_complete_demo.py", "--output-dir", str(output / "demo")])
    run("audit", ["scripts/audit_algorithm_readiness.py", "--raw", str(args.raw),
                  "--predictions", str(oof), "--training-report", str(training),
                  "--output", str(output / "audit")])
    print(f"Open {output / 'audit' / 'index.html'}")


if __name__ == "__main__":
    main()
