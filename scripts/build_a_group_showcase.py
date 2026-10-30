"""Assemble one offline chart/data collection from immutable A-group evidence."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from string import Template

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fogguard.selected_fog import sha

LABELS = {"ankle_history_rf": "Ankle-history RF", "bilateral_rf": "Bilateral RF", "ankle_cnn": "Ankle CNN"}
BLUE, GOLD, GREEN = "#396b9e", "#d69843", "#41a18b"
FIGURES = [
    ("01_workflow", "Workflow and model roles", "methods.csv", "Implementation",
     "Complete IMU windows feed independent FoG and current-direction models. The bilateral RF is the fixed fusion reference."),
    ("02_evaluation_scope", "Participants and scoring pools", "evaluation_scope.csv", "TEST scope",
     "The same held-out people support both tasks. FoG and direction use explicitly different scoring pools."),
    ("03_fog_test", "Three FoG methods", "fog_test.csv", "TEST",
     "Higher RF recall comes with more false alarms. The CNN reduces false positives but misses more FoG windows."),
    ("04_direction_test", "Current direction and fusion", "direction_test.csv", "TEST",
     "Unknown predictions count as errors on known targets. Fixed fusion rejects more windows and reduces overall accuracy."),
    ("05_fog_fusion", "FoG before and after event confirmation", "fusion_test.csv", "TEST",
     "Fixed confirmation and hysteresis do not improve FoG F1 here; positive state can persist through score dips."),
    ("06_joint_timeline", "Synchronized TEST replay", "figure_selection.csv", "TEST example",
     "The first TEST record and its first 30 seconds are shown. False alarms and unknown direction remain visible."),
    ("07_fog_confusion", "FoG confusion matrices", "fog_test.csv", "TEST",
     "The same supported windows expose true positives, false alarms and missed FoG for all three methods."),
]


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def text_file(path, content):
    Path(path).write_bytes(content.encode("utf-8"))


def json_file(path, content):
    text_file(path, json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_manifest(folder):
    for name, digest in read(folder / "MANIFEST.json")["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Published evidence hash mismatch: {folder.name}/{name}")


def validate_counts(results):
    fog = results["a1_common_windows"]
    if len({(r["windows"], r["positive_windows"]) for r in fog}) != 1:
        raise ValueError("FoG methods do not share the stated scoring pool")
    for row in fog:
        if sum(row[k] for k in ("TP", "FP", "FN", "TN")) != row["windows"]:
            raise ValueError("FoG confusion counts disagree with denominator")
    raw = results["a2_current_direction"]
    known = raw["primary_known_targets"]
    fused = results["fusion"]["direction_on_a2_arrival_grid"]["primary_known_targets"]
    if (known["windows"] + raw["missing_sensor_windows"] + raw["unknown_target_windows"]
            != raw["total_grid_windows"] or known["windows"] != fused["windows"]):
        raise ValueError("Direction scoring pools disagree")
    for report in (known, fused):
        matrix = np.asarray(report["confusion_matrix"])
        if matrix.sum() != report["windows"]:
            raise ValueError("Direction confusion counts disagree with denominator")
        if not np.isclose(np.trace(matrix) / report["windows"], report["accuracy"]):
            raise ValueError("Direction accuracy is inconsistent with saved counts")


def save(fig, out, name):
    fig.savefig(out / "figures" / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(out / "figures" / f"{name}.svg", facecolor="white")
    plt.close(fig)


def data_scope_figure(out, lock, results):
    fog = results["a1_common_windows"][0]
    direction = results["a2_current_direction"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [0.85, 1.5]})
    counts = [len(lock["split"][name]) for name in ("train", "validation", "test")]
    bars = axes[0].barh(["TRAIN", "Development", "TEST"], counts,
                        color=[BLUE, "#bdcddd", GREEN], height=0.55)
    axes[0].bar_label(bars, padding=5)
    axes[0].set(xlim=(0, max(counts) + 3), xlabel="Participants", title="Participant-separated roles")
    axes[0].invert_yaxis()
    parts = [
        (0, fog["windows"] - fog["positive_windows"], "Non-FoG", "#b9d0e3"),
        (0, fog["positive_windows"], "FoG", BLUE),
        (1, direction["primary_known_targets"]["windows"], "Known direction", GREEN),
        (1, direction["unknown_target_windows"], "Unknown target", GOLD),
        (1, direction["missing_sensor_windows"], "Invalid sensor", "#bac3cf"),
    ]
    left = {0: 0, 1: 0}
    for y, width, label, color in parts:
        axes[1].barh(y, width, left=left[y], color=color, height=0.48, label=label)
        axes[1].text(left[y] + width / 2, y, f"{width:,}", ha="center", va="center", fontsize=10)
        left[y] += width
    axes[1].set(yticks=[0, 1], yticklabels=["FoG matched pool", "Direction grid"],
                xlabel="Windows", title="TEST denominators")
    axes[1].invert_yaxis()
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, fontsize=9)
    fig.suptitle("One TEST cohort, task-specific scoring pools", weight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save(fig, out, "02_evaluation_scope")


def fusion_figure(out, results):
    primary = results["primary_fusion_fog_method"]
    raw = next(r for r in results["a1_common_windows"] if r["method"] == primary)
    native = results["a1_native_support"][primary]
    stable = results["fusion"]["stable_fog_on_a1_arrival_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    x = np.arange(3)
    for shift, row, label, color in [(-0.18, raw, "Raw bilateral RF", BLUE),
                                     (0.18, stable, "Stable FoG event", GOLD)]:
        bars = axes[0].bar(x + shift, [row[k] for k in ("precision", "recall", "fog_f1")],
                           0.36, color=color, label=label)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    axes[0].set(xticks=x, xticklabels=["Precision", "Recall", "FoG F1"], ylim=(0, 1.14),
                title=f"Same {raw['windows']:,} arrival windows")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=9)
    for ax, values, title, unit in [
        (axes[1], [native["false_alarm_s"], stable["false_alarm_s"]], "False alarm duration", "Seconds"),
        (axes[2], [100 * native["time_recall"], 100 * stable["time_recall"]], "Annotated FoG time recall", "Percent"),
    ]:
        bars = ax.bar(["Raw", "Stable"], values, color=[BLUE, GOLD], width=0.55)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.25))
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Fixed event aggregation: quantify the tradeoff", weight="bold", fontsize=15)
    fig.text(0.5, 0.015, "Window classification and causal raw-time overlap are different measures. "
             "Recovery is a model-score state, not measured forward movement.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    save(fig, out, "05_fog_fusion")


def table(headers, rows):
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                   for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def csv_file(out, name, rows):
    pd.DataFrame(rows).to_csv(out / "data" / name, index=False, lineterminator="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out, root = args.output_dir, Path(__file__).resolve().parents[1]
    if out.exists():
        parser.error("Use a fresh output directory; keep published evidence intact")
    evidence = root / "docs/a_group"
    verify_manifest(evidence)
    verify_manifest(root / "docs/a1/figures")
    verify_manifest(root / "docs/a2/fusion/presentation")
    results, lock = read(evidence / "results.json"), read(evidence / "TEST_LOCK.json")
    validate_counts(results)
    source_names = ["docs/a_group/results.json", "docs/a_group/TEST_LOCK.json",
                    "docs/a_group/MANIFEST.json", "docs/a1/figures/model_comparison.csv",
                    "docs/a2/results.json", "docs/a2/fusion/presentation/MANIFEST.json"]
    before = {name: sha(root / name) for name in source_names}
    for name in ("figures", "data"):
        (out / name).mkdir(parents=True, exist_ok=False)
    reused = {
        "01_workflow": root / "docs/a2/fusion/presentation/pipeline",
        "03_fog_test": evidence / "presentation/01_fog_test",
        "04_direction_test": evidence / "presentation/03_direction_test",
        "06_joint_timeline": evidence / "presentation/04_test_timeline",
        "07_fog_confusion": evidence / "presentation/02_fog_confusion",
    }
    copied = {}
    for name, source in reused.items():
        for suffix in (".png", ".svg"):
            src, dest = source.with_suffix(suffix), out / "figures" / (name + suffix)
            shutil.copyfile(src, dest)
            copied[dest.relative_to(out).as_posix()] = {
                "source": src.relative_to(root).as_posix(), "sha256": sha(src),
            }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "fogguard-showcase-v1"})
    data_scope_figure(out, lock, results)
    fusion_figure(out, results)
    fog = results["a1_common_windows"]
    current = results["a2_current_direction"]
    raw = current["primary_known_targets"]
    fused = results["fusion"]["direction_on_a2_arrival_grid"]["primary_known_targets"]
    primary = results["primary_fusion_fog_method"]
    methods = [
        {"module": "FoG", "method": LABELS[m], "input": sensors, "representation": features,
         "model": model, "decision_rule": f"FoG threshold {lock['fog_models'][m]['threshold']:.2f}"}
        for m, sensors, features, model in [
            ("ankle_history_rf", "Left ankle: acceleration + gyro", "140 current/history features", "300-tree RF"),
            ("bilateral_rf", "Both ankles: gyro", "66 window features", "300-tree RF; fusion reference"),
            ("ankle_cnn", "Left ankle: gyro", "120 x 3 centered waveform", "4 convolution layers; 8,353 parameters"),
        ]
    ] + [{"module": "Direction", "method": "Current-direction RF",
          "input": "Left ankle + back: acceleration + gyro", "representation": "12 gravity/rotation features",
          "model": "300-tree RF", "decision_rule": "Learned unknown class; fusion confidence gate 0.60"}]
    csv_file(out, "methods.csv", methods)
    scope = [{"partition": p, "measure": "participants", "value": len(lock["split"][p])}
             for p in ("train", "validation", "test")]
    scope += [{"partition": "test", "measure": name, "value": value} for name, value in [
        ("records", results["records"]), ("fog_matched_windows", fog[0]["windows"]),
        ("fog_positive_windows", fog[0]["positive_windows"]),
        ("direction_grid_windows", current["total_grid_windows"]),
        ("direction_known_valid_windows", raw["windows"]),
        ("direction_unknown_target_windows", current["unknown_target_windows"]),
        ("direction_invalid_sensor_windows", current["missing_sensor_windows"]),
    ]]
    csv_file(out, "evaluation_scope.csv", scope)
    scalar = lambda r: {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
    fog_rows = [{"partition": "test", **scalar(r)} for r in fog]
    direction_rows = [{"partition": "test", "stage": stage, **scalar(r)}
                      for stage, r in [("raw_current_direction", raw), ("fused_direction", fused)]]
    csv_file(out, "fog_test.csv", fog_rows)
    csv_file(out, "direction_test.csv", direction_rows)
    per_class = [{"partition": "test", "stage": stage, "class": name, **values}
                 for stage, r in [("raw_current_direction", raw), ("fused_direction", fused)]
                 for name, values in r["per_class"].items() if name != "unknown"]
    csv_file(out, "direction_per_class.csv", per_class)
    stable = results["fusion"]["stable_fog_on_a1_arrival_grid"]
    raw_primary = next(r for r in fog if r["method"] == primary)
    fusion_rows = [{"partition": "test", "stage": stage,
                    **{k: r[k] for k in ("windows", "precision", "recall", "fog_f1", "accuracy")},
                    "false_alarm_s": timing["false_alarm_s"], "time_recall": timing["time_recall"]}
                   for stage, r, timing in [("raw_bilateral_rf", raw_primary, results["a1_native_support"][primary]),
                                            ("stable_fog_event", stable, stable)]]
    csv_file(out, "fusion_test.csv", fusion_rows)
    selection = [{"partition": "test", "rule": "First TEST record by sorted identity; first 30 seconds",
                  "selection_fixed_before_scoring": True, "per_window_patient_csv_included": False}]
    csv_file(out, "figure_selection.csv", selection)
    dev = pd.read_csv(root / "docs/a1/figures/model_comparison.csv")
    dev_direction = read(root / "docs/a2/results.json")["results"]["current"]["primary_known_targets"]
    comparison = [{"task": "FoG", "method": LABELS[r["method"]], "metric": "FoG F1",
                   "development_windows": int(dev.loc[dev.method.eq(r["method"]), "windows"].iloc[0]),
                   "development_value": float(dev.loc[dev.method.eq(r["method"]), "fog_f1"].iloc[0]),
                   "test_windows": r["windows"], "test_value": r["fog_f1"]} for r in fog]
    comparison += [{"task": "Direction", "method": "Raw current-direction RF", "metric": "Accuracy",
                    "development_windows": dev_direction["windows"], "development_value": dev_direction["accuracy"],
                    "test_windows": raw["windows"], "test_value": raw["accuracy"]}]
    csv_file(out, "development_vs_test.csv", comparison)
    all_metrics = []
    for task, rows, denominator in [("FoG", fog_rows, "matched_windows"),
                                     ("Direction", direction_rows, "known_valid_windows"),
                                     ("FoG fusion", fusion_rows, "source_arrival_windows")]:
        for row in rows:
            for metric in ("accuracy", "precision", "recall", "fog_f1", "macro_f1",
                           "prediction_coverage", "accepted_accuracy", "false_alarm_s", "time_recall"):
                if metric in row:
                    all_metrics.append({"task": task, "partition": "test",
                                        "method_or_stage": row.get("method", row.get("stage")),
                                        "denominator": denominator, "windows": row["windows"],
                                        "metric": metric, "value": row[metric],
                                        "unit": "seconds" if metric == "false_alarm_s" else "fraction"})
    csv_file(out, "all_metrics.csv", all_metrics)
    catalog = [{"id": name, "title": title, "cohort_or_role": cohort,
                "png": f"figures/{name}.png", "svg": f"figures/{name}.svg",
                "data": f"data/{data}", "caption": caption}
               for name, title, data, cohort, caption in FIGURES]
    pd.DataFrame(catalog).to_csv(out / "FIGURE_CATALOG.csv", index=False, lineterminator="\n")
    def figure(name):
        row = next(r for r in catalog if r["id"] == name)
        return (f'<figure><img src="{row["png"]}" alt="{html.escape(row["title"])}" loading="lazy">'
                f'<figcaption><b>{html.escape(row["cohort_or_role"])}</b> · {html.escape(row["caption"])}'
                '<div class="figure-links">'
                f'<a href="{row["png"]}" download>PNG</a><a href="{row["svg"]}" download>Editable SVG</a>'
                f'<a href="{row["data"]}" download>{"Selection rule CSV" if name == "06_joint_timeline" else "Data CSV"}</a>'
                '</div></figcaption></figure>')
    values = {f"{key}_figure": figure(name) for key, name in [
        ("workflow", "01_workflow"), ("scope", "02_evaluation_scope"), ("fog", "03_fog_test"),
        ("direction", "04_direction_test"), ("fusion", "05_fog_fusion"),
        ("timeline", "06_joint_timeline"), ("confusion", "07_fog_confusion")]}
    values["scope_cards"] = "".join(f'<div class="stat"><strong>{number}</strong><span>{label}</span>'
                                    f'<small>{note}</small></div>' for number, label, note in [
        (f'{len(results["participants"])} / {results["records"]}', "TEST people / records", "Participant-held-out evaluation"),
        (f'{fog[0]["windows"]:,}', "Matched FoG windows", f'{fog[0]["positive_windows"]} positive windows'),
        (f'{raw["windows"]:,}', "Known valid direction windows", f'Of {current["total_grid_windows"]:,} grid windows'),
    ])
    values["methods_table"] = table(["Method", "Input", "Representation", "Model", "Decision rule"],
                                    [[r[k] for k in ("method", "input", "representation", "model", "decision_rule")]
                                     for r in methods])
    values["fog_table"] = table(["FoG method", "Accuracy", "Precision", "Recall", "FoG F1"],
                                [[LABELS[r["method"]], f'{r["accuracy"]:.2%}', f'{r["precision"]:.3f}',
                                  f'{r["recall"]:.3f}', f'{r["fog_f1"]:.3f}'] for r in fog])
    values["fog_note"] = (f'Always predicting non-FoG gives {fog[0]["always_non_fog_accuracy"]:.2%} accuracy '
                          'and zero FoG recall/F1. Read precision, recall and F1 together. '
                          'The bilateral RF remains the fusion reference selected before TEST.')
    values["direction_table"] = table(["Stage", "Accuracy", "Macro-F1", "Accepted coverage", "Accepted-only accuracy"],
        [[name, f'{r["accuracy"]:.2%}', f'{r["macro_f1"]:.3f}', f'{r["prediction_coverage"]:.2%}',
          f'{r["accepted_accuracy"]:.2%}'] for name, r in [("Raw current-direction RF", raw), ("Fixed fusion", fused)]])
    values["direction_note"] = (f'Both rows use {raw["windows"]} known-target windows. Always-straight accuracy '
                                f'is {raw["always_straight_accuracy"]:.2%}. Accepted-only accuracy excludes rejected '
                                'windows, so it cannot replace overall accuracy. Fusion does not improve overall accuracy.')
    values["fusion_table"] = table(["Stage", "Precision", "Recall", "FoG F1", "False alarm time", "FoG time recall"],
        [[r["stage"].replace("_", " "), f'{r["precision"]:.3f}', f'{r["recall"]:.3f}', f'{r["fog_f1"]:.3f}',
          f'{r["false_alarm_s"]:.1f} s', f'{r["time_recall"]:.2%}'] for r in fusion_rows])
    values["development_table"] = table(["Task / model", "Metric", "Development n", "Development", "TEST n", "TEST"],
        [[r["method"], r["metric"], r["development_windows"], f'{r["development_value"]:.3f}',
          r["test_windows"], f'{r["test_value"]:.3f}'] for r in comparison])
    values["figure_catalog"] = ('<table class="catalog"><thead><tr><th>Figure</th><th>Scope</th><th>Downloads</th>'
                                '</tr></thead><tbody>' + "".join(
        f'<tr><td>{html.escape(r["title"])}</td><td>{r["cohort_or_role"]}</td><td>'
        f'<a href="{r["png"]}" download>PNG</a> · <a href="{r["svg"]}" download>SVG</a> · '
        f'<a href="{r["data"]}" download>CSV</a></td></tr>' for r in catalog) + '</tbody></table>')
    values["data_links"] = "".join(f'<a href="data/{name}" download>{label}</a>' for name, label in [
        ("all_metrics.csv", "All TEST metrics"), ("direction_per_class.csv", "Direction by class"),
        ("development_vs_test.csv", "Development vs TEST"), ("methods.csv", "Method specification"),
        ("DATA_DICTIONARY.md", "Data dictionary"),
    ])
    template_dir = root / "src/fogguard/presentation"
    text_file(out / "index.html", Template((template_dir / "a_group_showcase.html").read_text("utf-8")).substitute(values))
    shutil.copyfile(template_dir / "a_group_showcase.css", out / "showcase.css")
    json_file(out / "data/summary.json", {"scope": {"test_participants": len(results["participants"]),
        "test_records": results["records"], "source_results_sha256": before["docs/a_group/results.json"]},
        "test_fog": fog_rows, "test_direction": direction_rows, "test_fog_fusion": fusion_rows,
        "methods": methods, "development_comparison": comparison})
    text_file(out / "data/DATA_DICTIONARY.md", "# Chart data\n\n"
              "All metric values in CSV/JSON are fractions in [0,1], except seconds, counts and thresholds. "
              "HTML formats accuracy and coverage as percentages.\n\n"
              "FoG comparison uses matched windows. Direction uses valid known-target windows; rejected "
              "predictions count as errors. Accepted-only accuracy excludes rejected predictions. "
              "Macro-F1 averages straight/left/right. `support` is per-class window count.\n\n"
              "Time recall uses causal raw-time overlap, not the window-classification denominator. "
              "The `windows` column in all_metrics.csv identifies the classification grid; it is not "
              "the time metric denominator.\n\n"
              "methods.csv describes implementation, evaluation_scope.csv gives cohort/pool counts, "
              "and figure_selection.csv records the fixed replay display rule. "
              "No per-window patient CSV is published. Development values are separately labeled.\n")
    text_file(out / "README.md", "# FoGGuard A-group unified charts and data\n\n"
              "Open index.html locally. This collection needs no server or internet. "
              "Figures are English PNG/SVG, with short explanations and aggregate CSV data.\n\n"
              "Order: workflow/methods, evaluation scope, FoG, current direction, FoG fusion, "
              "synchronized TEST replay. FoG confusion counts and development values are supporting material.\n\n"
              "TEST is the main effect report. Development is only method-selection context. "
              "Original model outputs, source results and the fixed replay selection are unchanged.\n\n"
              "See FIGURE_CATALOG.csv for figure/caption/data mapping and data/DATA_DICTIONARY.md for units.\n")
    if before != {name: sha(root / name) for name in source_names}:
        raise ValueError("Evidence changed while assembling the showcase")
    json_file(out / "MANIFEST.json", {"schema": "fogguard.a_group.showcase.v1",
        "source_sha256": {name: sha(root / name) for name in [
            "scripts/build_a_group_showcase.py", "src/fogguard/presentation/a_group_showcase.html",
            "src/fogguard/presentation/a_group_showcase.css"]},
        "evidence_sha256": before, "reused_figures": copied, "inputs_unchanged": True,
        "models_refitted": 0, "test_scores_recomputed": False,
        "files": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}})
    print(f"Saved {len(FIGURES)} figures, CSV data and one unified page to {out}")


if __name__ == "__main__":
    main()
