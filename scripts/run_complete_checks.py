"""One command: original tests, extension tests, lint, original evaluation, full demo."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/complete_checks"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    logs = output / "checks"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root / "src"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    commands = {
        "pytest": ["-m", "pytest", "-q"],
        "ruff": ["-m", "ruff", "check", "."],
        "original_evaluation": ["scripts/run_automation2_evaluation.py", "--output-dir",
                                str(output / "original_evaluation")],
        "complete_demo": ["scripts/run_complete_demo.py", "--output-dir", str(output / "demo"),
                          "--predictions", "docs/examples/robot2_predictions.csv"],
    }
    results = {}
    for name, command in commands.items():
        process = subprocess.run([sys.executable, "-X", "utf8", *command], check=False,
                                 cwd=root, env=env,
                                 capture_output=True, encoding="utf-8", errors="replace")
        (logs / f"{name}.txt").write_text(process.stdout + process.stderr, encoding="utf-8")
        results[name] = {"exit_code": process.returncode, "arguments": command}
        print(f"{name}: exit {process.returncode}\n{process.stdout[-800:]}", flush=True)
    (logs / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return int(any(row["exit_code"] for row in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
