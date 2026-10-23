"""Run tests, lint and evaluation with UTF-8 enabled on Windows and Ubuntu."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/automation2"))
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--training-report", type=Path)
    args = parser.parse_args()
    if bool(args.oof) != bool(args.training_report):
        parser.error("Supply --oof and --training-report together")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    logs = output / "checks"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"))
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    commands = {
        "pytest": ["-m", "pytest", "-q"],
        "ruff": ["-m", "ruff", "check", "."],
        "evaluation": ["scripts/run_automation2_evaluation.py", "--output-dir", str(output)],
        "review": ["scripts/build_automation2_review.py", "--input-dir", str(output)],
    }
    if args.oof:
        commands["evaluation"] += ["--oof", str(args.oof.resolve()),
                                   "--training-report", str(args.training_report.resolve())]
    results = {}
    for name, command in commands.items():
        if name == "review" and results["evaluation"]["exit_code"] != 0:
            results[name] = {"exit_code": 1, "status": "skipped: evaluation failed"}
            continue
        process = subprocess.run([sys.executable, "-X", "utf8", *command], check=False,
                                 cwd=root, env=env,
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        (logs / f"{name}.txt").write_text(process.stdout + process.stderr, encoding="utf-8")
        results[name] = {"exit_code": process.returncode, "arguments": command}
        print(f"{name}: exit={process.returncode}\n{process.stdout[-1500:]}", flush=True)
        if process.stderr:
            print(process.stderr[-1500:], flush=True)
    (logs / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return int(any(row["exit_code"] != 0 for row in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
