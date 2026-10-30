"""Build English figures and anonymous aggregate tables from a verified frozen run."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

STRATEGIES = ["population", "personal", "population_calibrated"]
LABELS = ["Population", "Personal", "Population + calibration"]
COLORS = ["#3869a8", "#1b937c", "#d58b36"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, output, name, caption):
    fig.text(.02, .015, caption, fontsize=9, color="#475569")
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output_dir
    result = json.loads((run / "results.json").read_text("utf-8"))
    verification = json.loads((run / "INDEPENDENT_VERIFICATION.json").read_text("utf-8"))
    if verification["status"] != "PASS":
        parser.error("Independent verification must pass first")
    manifest = json.loads((run / "MANIFEST.json").read_text("utf-8"))
    assert digest(run / "MANIFEST.json") == verification["run_manifest_sha256"]
    assert digest(run / "results.json") == manifest["files"]["results.json"]
    if out.exists():
        parser.error("Use a fresh presentation output directory")
    out.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.hashsalt": "fogguard-personal-v1"})
    pooled = result["pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    fig.subplots_adjust(left=.07, right=.98, top=.78, bottom=.23, wspace=.28)
    fig.suptitle("Personal training improves pooled F1, with more false alarms", fontsize=17, y=.96)
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("fog_f1", "FoG F1")]
    for i, s in enumerate(STRATEGIES):
        bars = axes[0].bar(np.arange(3) + (i - 1) * .25,
                           [pooled[s][m[0]] for m in metrics], width=.24,
                           color=COLORS[i], label=LABELS[i])
        axes[0].bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    axes[0].set(xticks=np.arange(3), xticklabels=[m[1] for m in metrics], ylim=(0, 1.03),
                title="Pooled window metrics | higher is better")
    bars = axes[1].bar(np.arange(3), [pooled[s]["false_positive_rate"] for s in STRATEGIES],
                       color=COLORS, width=.6)
    axes[1].bar_label(bars, labels=[f"{pooled[s]['false_positive_rate']:.1%}" for s in STRATEGIES], padding=4)
    axes[1].set(xticks=np.arange(3), xticklabels=["Population", "Personal", "Population +\ncalibration"],
                ylim=(0, .35), title="Non-FoG false-positive rate | lower is better")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .10),
               ncol=3, frameon=False)
    save(fig, out, "01_pooled_comparison",
         "Exploratory TRAIN-pool blocked CV. Same 8 people, 31 records, 4,090 windows (1,298 FoG). Original TEST excluded.")

    people = pd.DataFrame(result["per_person"])
    subjects = sorted(people.subject.unique())
    positive = people[people.strategy.eq("population")].set_index("subject").positive_windows
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4))
    fig.subplots_adjust(left=.07, right=.98, top=.87, bottom=.15, hspace=.46)
    fig.suptitle("Benefits vary by person: personal F1 improves for 4 of 8", fontsize=17, y=.96)
    for ax, metric, title, limit in zip(axes, ["fog_f1", "false_positive_rate"],
                                       ["FoG F1 | higher is better", "False-positive rate | lower is better"],
                                       [1, .65], strict=True):
        for i, s in enumerate(STRATEGIES):
            values = people[people.strategy.eq(s)].set_index("subject").loc[subjects, metric]
            ax.bar(np.arange(len(subjects)) + (i - 1) * .25, values, width=.24, color=COLORS[i], label=LABELS[i])
        ax.set(xticks=np.arange(len(subjects)),
               xticklabels=[f"{s}\nFoG n={positive[s]}" for s in subjects], ylim=(0, limit), title=title)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .06),
               ncol=3, frameon=False)
    save(fig, out, "02_participant_comparison",
         "Anonymous aliases. FoG n counts overlapping positive windows, not independent events. Small positive counts limit interpretation.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    fig.subplots_adjust(left=.15, right=.98, top=.76, bottom=.25, wspace=.55)
    fig.suptitle("Feasibility and evaluation scope limit the conclusion", fontsize=17, y=.96)
    axes[0].barh(["Eligible", "Too few adaptation\nFoG windows", "No supported\nevaluation windows"],
                 [31, 19, 8], color=[COLORS[1], "#94a3b8", "#cbd5e1"])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Outer folds", title="31 / 58 folds eligible")
    for y, n in enumerate([31, 19, 8]):
        axes[0].text(n + .6, y, str(n), va="center")
    axes[0].set_xlim(0, 37)
    for ax, split, title in zip(axes[1:], ["record_held_out", "session_held_out"],
                                ["Record holdout\n7 people | 3,992 windows", "Session holdout\n1 person | 98 windows"], strict=True):
        rows = {r["strategy"]: r for r in result["per_split_kind"] if r["split_kind"] == split}
        bars = ax.bar(np.arange(3), [rows[s]["fog_f1"] for s in STRATEGIES], color=COLORS)
        ax.bar_label(bars, fmt="%.3f", fontsize=10, padding=3)
        ax.set(ylim=(0, 1), title=title, ylabel="FoG F1", xticks=np.arange(3), xticklabels=["Pop.", "Personal", "Calib."])
    fig.text(.40, .13, "Record holdout is within-session. Session IDs are not verified dates.\nOnly 9 positive windows in session holdout; no across-day benefit established.", fontsize=10)
    save(fig, out, "03_feasibility_and_splits",
         "14 TRAIN participants / 70 records considered; 8 participants support paired comparison. Exclusions fixed from adaptation support.")

    columns = ["windows", "positive_windows", "precision", "recall", "fog_f1", "false_positive_rate", "TP", "FP", "FN", "TN",
               "false_alarm_s", "observed_s", "false_alarm_s_per_observed_minute"]
    comparison = pd.DataFrame([{"strategy": s, **{k: pooled[s][k] for k in columns},
                                **result["participant_macro"][s]} for s in STRATEGIES])
    comparison.to_csv(out / "comparison.csv", index=False)
    people.to_csv(out / "per_person.csv", index=False)
    pd.DataFrame(result["per_split_kind"]).to_csv(out / "per_split_kind.csv", index=False)
    pd.DataFrame(result["negative_only_population_controls"]).to_csv(out / "negative_only_controls.csv", index=False)
    pd.DataFrame([{"status": k, "folds": v} for k, v in result["fold_status"].items()]).to_csv(out / "feasibility.csv", index=False)
    for name in ["results.json", "INDEPENDENT_VERIFICATION.json"]:
        shutil.copyfile(run / name, out / name)
    shutil.copyfile(run / "PLAN_LOCK.json", out / "PROTOCOL_LOCK.json")
    pretty = comparison[["strategy", "precision", "recall", "fog_f1", "false_positive_rate",
                         "mean_participant_fog_f1", "mean_participant_recall", "false_alarm_s_per_observed_minute"]].copy()
    pretty.columns = ["Strategy", "Precision", "Recall", "FoG F1", "False-positive rate", "Person-macro F1", "Person-macro recall", "False alarm s / observed min"]
    table = pretty.to_html(index=False, float_format=lambda value: f"{value:.3f}", border=0)
    names = ["01_pooled_comparison", "02_participant_comparison", "03_feasibility_and_splits"]
    captions = ["Pooled F1 rises 0.689 to 0.735; precision falls and false-positive rate rises 17.4% to 24.0%. Calibration does not improve F1.",
                "Four participants improve and four worsen. Participant-macro recall falls from 0.658 to 0.621 despite the higher pooled recall.",
                "Six of fourteen participants cannot support the paired personal study. Across-session evidence is restricted to one person and nine positive windows."]
    cards = "".join(f'<section><img src="{n}.png" alt="{html.escape(c)}"><p>{c}</p><p><a href="{n}.png">PNG</a> · <a href="{n}.svg">SVG</a></p></section>' for n, c in zip(names, captions, strict=True))
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FoGGuard | Personalized FoG study</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f4f7fb;color:#172b46;margin:0}main{max-width:1280px;margin:auto;padding:32px 24px}
h1{font-size:34px;line-height:1.2}h2{font-size:24px}section,.box{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:24px;margin:24px 0}
img{width:100%;height:auto}a{color:#205da5}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px;text-align:right;border-bottom:1px solid #dbe3ed}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}.tag{color:#52677d}
</style><main><p class="tag">FoGGuard · Optional A1 research · 30 October 2026</p>
<h1>Does personal training improve FoG detection?</h1>
<p>It improves pooled F1 in this experiment, with more false alarms and uneven benefits across people.</p>
<div class="box"><strong>Scope: exploratory blocked cross-validation within the original TRAIN pool.</strong>
<p>14 participants / 70 records considered. The paired comparison covers 8 participants / 31 records / 4,090 overlapping windows, including 1,298 FoG windows. Original development and TEST participants are excluded. These scores cannot be compared directly with the original locked TEST F1.</p></div>
<section><h2>Implementation</h2><p>All strategies use the existing bilateral gyro representation: 120 samples at 60 Hz, step 30, 66 features, and a 300-tree random forest. Population RF fits other TRAIN participants, excluding the target. Personal RF fits target-person adaptation records only. Both use fixed threshold 0.23. Population + calibration retains population weights and chooses a threshold on the target adaptation records alone (grid 0.05–0.95).</p>
<p>Participants with multiple sessions use session holdout; others use whole-record holdout. Whole, disjoint records define adaptation and evaluation; windows never cross that boundary. At least 10 positive and 10 negative adaptation windows are required; eligibility is never based on held-out labels or scores. Personal fitting and threshold calibration use the same adaptation-label budget, in different ways.</p>
<p>Threshold 0.23 is inherited from prior TRAIN-based design. Target exclusion refers to the current population classifier fit; this is not a newly isolated benchmark with no prior knowledge of the TRAIN pool.</p></section>
""" + cards + '<section><h2>Exact aggregate results</h2><div class="scroll">' + table + """</div>
<p>Participant macro gives each person equal weight. Pooled metrics give each window equal weight. False alarm duration uses causal prediction availability, held for at most 0.5 s; its denominator is the same 2,097.78 observed seconds. It is not alarms per walking minute or measured clinical burden.</p></section>
<section><h2>Interpretation and limits</h2><p>Personal training is a promising optional adaptation strategy, not a universal replacement. The result does not prove that inter-person gait variation caused the original errors. Model data volume, same-session similarity and the fixed operating point also differ. No threshold retuning followed evaluation.</p>
<p>Record holdout is within-session; session IDs are grouping keys, not verified dates. Overlapping windows are not independent subjects. Only one participant supports session holdout, where personal F1 is lower. A prospective across-day or new-person trial is still needed for deployment claims.</p>
<p>Five participants with no positive supported windows are retained as separate population-only controls (CSV below). FoG recall/F1 are not evaluable for those controls; lack of a positive supported window does not establish that a person never freezes.</p>
<p>The original three delivered methods, weights, locked TEST results, A2 fusion and A-group showcase are preserved. This study does not replace the main pipeline.</p></section>
<section><h2>Data and reproducibility</h2><p>
<a href="comparison.csv">Aggregate comparison CSV</a> · <a href="per_person.csv">Per-person CSV</a> ·
<a href="per_split_kind.csv">Split-type CSV</a> · <a href="feasibility.csv">Feasibility CSV</a> ·
<a href="negative_only_controls.csv">Negative-only control CSV</a> · <a href="results.json">Full aggregate JSON</a> ·
<a href="PROTOCOL_LOCK.json">Pre-fit protocol lock</a> · <a href="INDEPENDENT_VERIFICATION.json">Independent verification</a></p>
<p>12,270 saved-model predictions replayed; paired metric counts, target exclusion, adaptation-only threshold selection and original evidence hashes verified. Public files contain anonymous aggregates, not raw signals or model weights.</p></section></main></html>"""
    (out / "index.html").write_bytes(page.encode("utf-8"))
    files = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    package = {"schema": "fogguard.personalized_fog.presentation.v1", "files": files,
               "builder_sha256": digest(__file__), "source_results_sha256": digest(run / "results.json"),
               "scope": result["scope"], "browser_render_check": "NOT_RUN; offline link/image/data verification only"}
    (out / "MANIFEST.json").write_bytes((json.dumps(package, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"figures": len(names), "csv_tables": 5, "files": len(files)}))


if __name__ == "__main__":
    main()
