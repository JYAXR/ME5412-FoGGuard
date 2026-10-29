"""Fuse saved A1/A2 runtime predictions without labels, training or feedback."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from fogguard.perception import fuse_perception, summarize_perception
from fogguard.selected_fog import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fog", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--clock", type=Path, help="Optional record keys + timestamp_s CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    result = fuse_perception(pd.read_csv(args.fog), pd.read_csv(args.direction),
                             clock=pd.read_csv(args.clock) if args.clock else None)
    summary = summarize_perception(result)
    inputs = {name: sha(path) for name, path in {
        "fog": args.fog, "direction": args.direction, "clock": args.clock,
    }.items() if path is not None}
    args.output_dir.mkdir(parents=True)
    runtime = args.output_dir / "perception_runtime.csv"
    result.to_csv(runtime, index=False)
    write_json(args.output_dir / "summary.json", summary)
    root = Path(__file__).resolve().parents[1]
    write_json(args.output_dir / "MANIFEST.json", {
        "schema": "fogguard.perception.manifest.v1", "inputs_sha256": inputs,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                               text=True).strip(),
        "source_sha256": {name: sha(root / name) for name in (
            "scripts/fuse_perception.py", "src/fogguard/perception.py",
            "src/fogguard/context/fogstar.py",
        )},
        "files": {name: sha(args.output_dir / name)
                  for name in ("perception_runtime.csv", "summary.json")},
        "scope": "Runtime-only fusion; truth not used; no feedback or cue commands generated",
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
