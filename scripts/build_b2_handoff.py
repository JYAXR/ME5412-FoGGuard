"""Create/verify a private standard B2 bundle from saved updated cue outputs."""

import argparse
import json
from pathlib import Path

from fogguard.evaluation.handoff_bundle import build_bundle, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps({"status": "PASS", "files_verified": verify_manifest(args.verify)}))
    elif args.run and args.output:
        print(json.dumps(build_bundle(args.run, args.output, plan=args.plan)))
    else:
        parser.error("Use --verify BUNDLE or --run RUN --output NEW_DIRECTORY [--plan PLAN]")


if __name__ == "__main__":
    main()
