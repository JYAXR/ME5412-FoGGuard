"""Evaluate frozen A-group models on repeatable variants of validation sensor clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from evaluate_b2_randomized_models import check_plan, load_frozen, valid_source_windows

from fogguard.evaluation.randomized_model_scoring import (
    attach_to_truth,
    direction_metrics,
    fog_metrics,
)
from fogguard.evaluation.randomized_plan import file_hash
from fogguard.evaluation.sensor_variants import generate_variant

ROOT = Path(__file__).resolve().parents[1]


def run(plan, package, clean_run, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    check_plan(plan)
    old = json.loads((clean_run/"RESULTS.json").read_text(encoding="utf8"))
    if old["sample_plan_sha256"] != file_hash(plan/"PLAN.json"):
        raise ValueError("Baseline was generated from a different plan")
    torch.set_num_threads(1)
    models, model_hashes = load_frozen(package/"fog-selected", package/"a2-current/current/model.joblib")
    if model_hashes != old["model_hashes"]:
        raise ValueError("Baseline model hashes differ")
    output.mkdir(parents=True)
    for folder in ("generated_sensors", "runtime"):
        (output/folder).mkdir()
    samples = pd.read_csv(plan/"sample_manifest.csv")
    rows, paired_rows, params = [], [], {}
    for idx, sample in enumerate(samples.itertuples(index=False)):
        sensors = pd.read_csv(plan/"clips"/f"{sample.clip}_sensors.csv")
        truth = pd.read_csv(plan/"clips"/f"{sample.clip}_window_truth.csv")
        baselines = {}
        for name in models:
            basepath = clean_run/"runtime"/f"{sample.clip}_clean_{name}.csv"
            if not basepath.exists():
                raise ValueError(f"Missing baseline runtime {basepath}")
            base = pd.read_csv(basepath)
            baselines[name], outside = attach_to_truth(truth, base, kind="direction" if name == "current_direction" else "fog")
            if outside:
                raise ValueError("Baseline predictions outside scoring grid")
        for draw in range(3):
            seed = 20261103 + idx * 1009 + draw * 131
            sensor, param = generate_variant(sensors, seed)
            key = f"{sample.clip}_v{draw+1}"
            params[key] = {"seed": seed, "channels": param}
            sensor.to_csv(output/"generated_sensors"/f"{key}.csv", index=False)
            for name, model in models.items():
                runtime = model.predict_frame(sensor)
                if valid_source_windows(sensor, runtime, model.channels if name == "current_direction" else model.channel_names):
                    raise ValueError(f"Invalid source windows: {key} {name}")
                runtime.to_csv(output/"runtime"/f"{key}_{name}.csv", index=False)
                attached, outside = attach_to_truth(truth, runtime, kind="direction" if name == "current_direction" else "fog")
                if outside:
                    raise ValueError("Generated prediction outside scoring grid")
                base = baselines[name]
                if not base.supported.equals(attached.supported):
                    raise ValueError(f"Unexpected coverage change: {key} {name}")
                attached["clip"], attached["variant"], attached["model"] = sample.clip, draw+1, name
                rows.append(attached)
                mask = base.supported & attached.supported
                if name == "current_direction":
                    mask &= attached.truth_direction.ne("unknown")
                paired_rows.append({"clip": sample.clip, "variant": draw+1, "model": name,
                                    "paired_windows": int(mask.sum()),
                                    "changed_predictions": int((base.loc[mask, "predicted"].to_numpy() !=
                                                                attached.loc[mask, "predicted"].to_numpy()).sum())})
        print(f"Completed {idx+1}/{len(samples)} clips", flush=True)
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(output/"scored_windows.csv", index=False)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output/"paired_comparisons.csv", index=False)
    metrics = []
    for (name, variant), group in frame.groupby(["model", "variant"]):
        metric = direction_metrics(group) if name == "current_direction" else fog_metrics(group)
        changes = paired.loc[paired.model.eq(name) & paired.variant.eq(variant)]
        metrics.append({"model": name, "variant": int(variant), **metric,
                        "paired_windows": int(changes.paired_windows.sum()),
                        "changed_predictions": int(changes.changed_predictions.sum())})
    summary = {"schema":"fogguard.b2.sensor_variants.v1", "clips":len(samples), "variants_per_clip":3,
               "generated_clip_count":len(samples)*3, "model_runs":len(samples)*3*len(models),
               "plan_sha256":file_hash(plan/"PLAN.json"), "baseline_results_sha256":file_hash(clean_run/"RESULTS.json"),
               "models":model_hashes, "generator_sha256":file_hash(ROOT/"src/fogguard/evaluation/sensor_variants.py"),
               "script_sha256":file_hash(ROOT/"scripts/evaluate_sensor_variants.py"),
               "generation":"Per-channel gain uniform 0.97–1.03; offset uniform +/-0.003 g or +/-0.3 deg/s; Gaussian noise SD 0.005 g or 0.5 deg/s",
               "limitations":["Generated by perturbing real development-validation clips; not independent patients or newly collected data",
                              "Transferred source labels are assumed, not independently re-annotated",
                              "Perturbations are engineering assumptions, not validated physiological or sensor distributions",
                              "Repeated overlapping windows and variants are correlated; no independent-sample claims",
                              "No model retraining or threshold changes"], "metrics":metrics}
    (output/"parameters.json").write_text(json.dumps(params,ensure_ascii=False,indent=2), encoding="utf8")
    (output/"RESULTS.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False), encoding="utf8")
    print(json.dumps({"clips":len(samples),"generated":len(samples)*3,"runs":len(samples)*3*len(models)},ensure_ascii=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan",type=Path,required=True)
    ap.add_argument("--package",type=Path,required=True)
    ap.add_argument("--clean-run",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args = ap.parse_args()
    run(args.plan,args.package,args.clean_run,args.output)

