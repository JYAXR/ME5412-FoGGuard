"""One-command verification of the cumulative Automation 2 delivery."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--training-report", type=Path)
    args = parser.parse_args()
    supplied = sum(path is not None for path in (args.raw, args.oof, args.training_report))
    if supplied not in (0, 3):
        parser.error("Provide --raw, --oof, --training-report together, or omit all three")
    root = Path(__file__).resolve().parents[1]
    output = (args.output_dir.resolve() if args.output_dir else root / "reports" /
              "automation2-delivery" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; use a fresh run directory to preserve evidence")
    logs = output / "checks"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    commands = {
        "pytest": ["-m", "pytest", "tests", "recognition", "-q"],
        "ruff": ["-m", "ruff", "check", "."],
        "baseline_evaluation": ["scripts/run_automation2_evaluation.py", "--output-dir",
                                str(output / "baseline")],
        "integrated_demo": ["scripts/run_complete_demo.py", "--output-dir",
                            str(output / "integrated"), "--predictions",
                            "docs/examples/robot2_predictions.csv"],
    }
    if supplied:
        commands["real_data_audit"] = ["recognition/evaluate.py", "--raw", str(args.raw.resolve()),
                                      "--predictions", str(args.oof.resolve()), "--training-report",
                                      str(args.training_report.resolve()), "--output",
                                      str(output / "recognition")]
    results = {}
    for name, arguments in commands.items():
        process = subprocess.run([sys.executable, "-X", "utf8", *arguments], cwd=root,
                                 env=env, capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", check=False)
        (logs / f"{name}.txt").write_text(process.stdout+process.stderr, encoding="utf-8")
        results[name] = {"exit_code": process.returncode}
        display = process.stdout[-2000:]
        if name == "real_data_audit" and process.returncode == 0:
            report = json.loads((output / "recognition" / "核验与指标.json").read_text("utf-8"))
            display = json.dumps({"raw_records": report["records"],
                                  "records_with_predictions": report["records_with_predictions"],
                                  "validated_raw_windows": report["validated_raw_windows"],
                                  "window_metrics": report["metrics"]}, indent=2)
        print(f"{name}: exit={process.returncode}\n{display}", flush=True)
        if process.stderr:
            print(process.stderr[-2000:], flush=True)
    if not supplied:
        results["real_data_audit"] = {"status": "NOT_RUN", "reason": "Raw data and OOF inputs omitted"}
    summary = {"schema": "fogguard.automation2.checks.v1", "checks": results,
               "real_data_retrained": False}
    (logs / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return int(any(result.get("exit_code", 0) for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
