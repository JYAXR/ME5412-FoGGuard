"""Evaluate Robot 2 temporal integration on held-out FoG-STAR predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fogguard.evaluation import evaluate_fogstar_integration
from fogguard.fogstar_pipeline import integrate_fogstar_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/fogstar_oof_predictions.csv"),
    )
    parser.add_argument(
        "--integrated-output",
        type=Path,
        default=Path("reports/fogstar_robot2_end_to_end.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/fogstar_robot2_integration.json"),
    )
    args = parser.parse_args()

    integrated = integrate_fogstar_predictions(pd.read_csv(args.predictions))
    report = evaluate_fogstar_integration(integrated)
    args.integrated_output.parent.mkdir(parents=True, exist_ok=True)
    integrated.to_csv(args.integrated_output, index=False)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {key: value for key, value in report.items() if key != "events"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.integrated_output}")
    print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
