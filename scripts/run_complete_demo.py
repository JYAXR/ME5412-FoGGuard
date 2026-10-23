"""Generate reproducible integrated demos and comparisons without model downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import pandas as pd

from fogguard.complete import complete_cases, run_complete_pipeline, strategy_configs
from fogguard.cueing.march import MarchConfig, MarchGuidanceController
from fogguard.cueing.replay import replay_guidance, summarize_guidance
from fogguard.evaluation.automation2 import audit_commands, evaluate_timeline


def generate(output, predictions_path=None, responses_path=None, base_config=None):
    output.mkdir(parents=True, exist_ok=True)
    strategies = strategy_configs(base_config)
    cases = complete_cases()
    if predictions_path:
        responses = pd.read_csv(responses_path) if responses_path else None
        events, _ = run_complete_pipeline(pd.read_csv(predictions_path), responses=responses)
        cases["external_predictions"] = (events, responses)
    results = []
    playback = []
    failures = []
    for name, (events, responses) in cases.items():
        events.to_csv(output / f"{name}_events.csv", index=False)
        if responses is not None:
            responses.to_csv(output / f"{name}_responses.csv", index=False)
        for strategy, config in strategies.items():
            commands = replay_guidance(events, config=config, responses=responses,
                                       controller_type=MarchGuidanceController)
            commands.to_csv(output / f"{name}_{strategy}.csv", index=False)
            errors = audit_commands(commands)
            failures.extend({"case": name, "strategy": strategy, **error} for error in errors)
            metrics = (evaluate_timeline(events, commands)
                       if {"scenario_id", "truth_fog_label"}.issubset(events.columns)
                       else {"available": False, "reason": "Independent truth labels not supplied"})
            summary = summarize_guidance(commands)
            results.append({"case": name, "strategy": strategy, "config": config.as_dict(),
                            "summary": summary, "metrics": metrics})
            rows = json.loads(commands.to_json(orient="records"))
            playback.append({"id": f"{name}/{strategy}", "case": name, "strategy": strategy,
                             "rows": rows, "summary": summary})
    manifest = {
        "schema": "fogguard-complete-demo-v1", "python": platform.python_version(),
        "scope": "Engineering simulation; constructed candidates and simulated responses.",
        "cases": len(cases), "runs": len(results), "audit_failures": failures,
        "results": results,
        "external_inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (predictions_path, responses_path) if path is not None
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    (output / "playback.json").write_text(json.dumps(playback, ensure_ascii=False), encoding="utf-8")
    comparisons = [{"case": row["case"], "strategy": row["strategy"],
                    **{key: value for key, value in row["summary"].items()
                       if isinstance(value, (int, float, str))}}
                   for row in results]
    pd.DataFrame(comparisons).to_csv(output / "comparison.csv", index=False)
    hashes = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.glob("*.csv"))}
    (output / "data_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    from fogguard.presentation.complete_review import write_complete_review
    write_complete_review(output / "index.html", playback, manifest)
    if failures:
        raise RuntimeError(f"Command lifecycle failures: {len(failures)}")
    print(f"Generated {len(cases)} cases / {len(results)} strategy runs: {output}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/complete"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.responses and not args.predictions:
        parser.error("--responses requires --predictions")
    generate(args.output_dir, args.predictions, args.responses,
             MarchConfig.load(args.config) if args.config else None)


if __name__ == "__main__":
    main()
