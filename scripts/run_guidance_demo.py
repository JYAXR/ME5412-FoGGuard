"""Replay visual/audio guidance without model downloads or network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from fogguard.cueing.contracts import GuidanceConfig
from fogguard.cueing.replay import replay_guidance, summarize_guidance
from fogguard.cueing.scenarios import build_guidance_cases
from fogguard.data.responses import load_step_responses
from fogguard.fogstar_pipeline import integrate_fogstar_predictions


def run_case(name, events, responses, config, output_dir):
    result = replay_guidance(events, config=config, responses=responses)
    fixed_config = replace(config, enable_response_policy=False, enable_fading=False,
                           enable_adaptive_spacing=False)
    fixed = replay_guidance(events, config=fixed_config, responses=responses)
    result.to_csv(output_dir / f"guidance_{name}.csv", index=False)
    fixed.to_csv(output_dir / f"guidance_{name}_fixed.csv", index=False)
    records = json.loads(result.to_json(orient="records"))
    (output_dir / f"guidance_{name}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    summary = summarize_guidance(result)
    summary["event_input_sha256"] = hashlib.sha256(
        events.to_csv(index=False).encode()
    ).hexdigest()
    summary["response_input_sha256"] = (
        hashlib.sha256(responses.to_csv(index=False).encode()).hexdigest()
        if responses is not None else None
    )
    summary["fixed_reference"] = summarize_guidance(fixed)
    summary["fixed_reference_config"] = fixed_config.as_dict()
    summary["single_factor_references"] = {}
    if config.enable_response_policy:
        for setting, suffix in (("enable_fading", "no_fade"),
                                ("enable_adaptive_spacing", "no_spacing_adjustment")):
            if not getattr(config, setting):
                continue
            reference_config = replace(config, **{setting: False})
            reference = replay_guidance(events, config=reference_config, responses=responses)
            filename = f"guidance_{name}_{suffix}.csv"
            reference.to_csv(output_dir / filename, index=False)
            summary["single_factor_references"][suffix] = {
                "changed_setting": setting, "effective_config": reference_config.as_dict(),
                "timeline_file": filename, "summary": summarize_guidance(reference),
            }
    if "cue" in events:
        summary["robot2_baseline_pulse_commands"] = int(events.cue.ne("none").sum())
    summary["source"] = (
        "constructed strategy inputs" if "source_dataset" in events
        and events.source_dataset.eq("constructed_guidance_case").all()
        else "supplied timeline; see original provenance"
    )
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scenario", default="all", help="all or a constructed acceptance case")
    source.add_argument("--predictions", type=Path, help="Robot 2 raw candidate CSV")
    source.add_argument("--events", type=Path, help="Already integrated Robot 2 event CSV")
    parser.add_argument("--responses", type=Path, help="Explicitly simulated responses by scenario_id")
    parser.add_argument("--config", type=Path, default=Path("config/guidance.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--adaptive-spacing", action="store_true")
    args = parser.parse_args()
    if args.fixed_only and args.adaptive_spacing:
        parser.error("--fixed-only and --adaptive-spacing are separate comparisons")
    config = GuidanceConfig.load(args.config)
    if args.fixed_only:
        config = replace(config, enable_response_policy=False, enable_fading=False,
                         enable_adaptive_spacing=False)
    if args.adaptive_spacing:
        config = replace(config, enable_adaptive_spacing=True)
    responses = load_step_responses(args.responses) if args.responses else None
    if args.events or args.predictions:
        if args.events:
            events = pd.read_csv(args.events)
        else:
            events = integrate_fogstar_predictions(pd.read_csv(args.predictions))
        cases = {"supplied": (events, responses)}
    else:
        provided = responses if responses is not None else load_step_responses(
            "data/demo/simulated_step_responses.csv"
        )
        cases = build_guidance_cases(provided)
        if args.scenario != "all":
            if args.scenario not in cases:
                parser.error(f"Unknown scenario. Choose: {', '.join(cases)}")
            cases = {args.scenario: cases[args.scenario]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False,
    )
    code = hashlib.sha256()
    for path in sorted((repository / "src/fogguard/cueing").glob("*.py")) + [Path(__file__)]:
        code.update(path.name.encode())
        code.update(path.read_bytes())
    report = {"schema_version": "guidance-proposal-v1", "effective_config": config.as_dict(),
              "base_commit": commit.stdout.strip() if commit.returncode == 0 else None,
              "guidance_code_sha256": code.hexdigest(),
              "runtime_versions": {name: version(name) for name in ("numpy", "pandas", "PyYAML")},
              "cases": {}}
    review = {}
    for name, (events, case_responses) in cases.items():
        review[name], report["cases"][name] = run_case(
            name, events, case_responses, config, args.output_dir,
        )
        summary = report["cases"][name]
        print(f"{name}: starts={summary['start_commands']}, stops={summary['stop_commands']}, "
              f"states={','.join(summary['states'])}")
    (args.output_dir / "guidance_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    # Display-only renderer consumes saved controller output; it never re-runs policy.
    from fogguard.cueing.review import write_guidance_review

    write_guidance_review(args.output_dir / "guidance_review.html", review, report)
    print(f"Review: {args.output_dir / 'guidance_review.html'}")


if __name__ == "__main__":
    main()
