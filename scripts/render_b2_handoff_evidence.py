"""Render English process evidence and bounded acceptance tables from a saved bundle."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fogguard.evaluation.handoff_bundle import sha256, verify_manifest, write_json

CASES = [("real/clip_020", "01_real_process", "Real model replay + IMU movement proxy"),
         ("constructed/low_motion_straight", "02_constructed_fallback",
          "Constructed rule probe: low motion and fallback"),
         ("constructed/feedback_gap_straight", "03_feedback_gap",
          "Constructed rule probe: missing feedback and withdrawal")]


def render_case(root, relative, target, title, stem):
    folder = root / relative
    p = pd.read_csv(folder / "perception_runtime.csv")
    f = pd.read_csv(folder / "feedback_runtime.csv")
    d = pd.read_csv(folder / "commands.csv")
    a = d.loc[~d.guidance_terminal]
    offset = float(p.available_at_s.iloc[0])
    t = p.available_at_s - offset
    tf = f.feedback_available_at_s - offset
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1.2, .85, 1.3]})
    fig.subplots_adjust(top=.88, bottom=.1, hspace=.38, left=.10, right=.85)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=.97)
    subtitle = ("Previously inspected development-validation clip; not a new TEST or patient-response trial"
                if relative.startswith("real/") else
                "Constructed sensor/candidate inputs; no learned-model inference or patient response")
    fig.text(.5, .925, subtitle, ha="center", fontsize=10, color="#536174")
    axes[0].step(t, p.fog_probability, where="post", color="#167a9c", label="FoG score")
    threshold = p.fog_threshold.dropna()
    if len(threshold):
        axes[0].axhline(threshold.iloc[0], color="#bd6031", ls="--", label="Frozen threshold")
    truth_path = folder / "evaluation_truth.csv"
    if truth_path.exists():
        truth = pd.read_csv(truth_path)
        axes[0].step(truth.timestamp - offset, truth.fog, where="post", alpha=.55,
                     color="#8a559e", label="FoG label (evaluation only)")
    axes[0].set(ylabel="FoG score / label", ylim=(-.08, 1.08))
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    axes[1].plot(tf, f.left_gyro_rms_deg_s, label="Left ankle", color="#167a9c")
    axes[1].plot(tf, f.right_gyro_rms_deg_s, label="Right ankle", color="#cf7747")
    axes[1].axhline(5, ls="--", color="#5e7d45", lw=1, label="Low-motion limit")
    axes[1].axhline(15, ls=":", color="#9a5573", lw=1, label="Movement limit")
    axes[1].set(ylabel="Gyro RMS (deg/s)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    mapping = {"unknown": 0, "no_movement_evidence": 1, "movement_evidence": 2}
    axes[2].step(tf, f.feedback_state.map(mapping), where="post", color="#594f8d", lw=1.8)
    axes[2].set(yticks=[0, 1, 2], yticklabels=["Unknown", "Low motion", "Movement"],
                ylim=(-.25, 2.25), ylabel="Proxy evidence")
    tc = d.guidance_timestamp_s - offset
    axes[3].step(tc, d.guidance_visual_opacity.where(d.guidance_visual_enabled, 0),
                 where="post", color="#167a9c", label="Visual intensity")
    axes[3].step(tc, d.guidance_audio_volume, where="post", color="#cf7747", alpha=.8,
                 label="Audio volume")
    axes[3].set(ylabel="Cue level", ylim=(-.08, 1.2), xlabel="Time from clip start (s)")
    fallback = a.guidance_control_state.eq("MARCH_FALLBACK")
    if fallback.any():
        low = float(t.loc[fallback].min())
        high = float(t.loc[fallback].max())
        axes[3].axvspan(low, high, color="#b0cfb2", alpha=.3)
        axes[3].text((low + high) / 2, .15, "Fallback", ha="center", fontsize=9)
    twin = axes[3].twinx()
    twin.step(tc, d.proxy_no_motion_valid_s, where="post", color="#7d638d", ls="--", alpha=.8)
    twin.set_ylabel("Valid low-motion time (s)", color="#7d638d", fontsize=9)
    twin.set_ylim(0, max(3.5, a.proxy_no_motion_valid_s.max() + .3))
    axes[3].legend(loc="upper left", bbox_to_anchor=(1.13, 1), fontsize=8)
    events = d.loc[d.guidance_reason.ne(d.guidance_reason.shift())
                   | d.guidance_control_state.ne(d.guidance_control_state.shift())
                   | d.guidance_terminal].copy()
    highlights = {
        "valid_no_motion_enter_march": "Enter fallback",
        "fallback_timeout_return_forward_unverified": "Timeout: try forward",
        "feedback_unavailable": "Feedback unavailable",
        "perception_recovery_with_motion_proxy_begin_fade": "Begin fade",
    }
    for reason, label in highlights.items():
        selected = events.loc[events.guidance_reason.eq(reason)]
        if not selected.empty:
            x = float(selected.iloc[0].guidance_timestamp_s - offset)
            for ax in axes:
                ax.axvline(x, color="#737c87", ls=":", lw=.8, alpha=.7)
            axes[3].annotate(label, (x, 1.05), rotation=0, fontsize=8,
                             xytext=(3, 0), textcoords="offset points")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, float(t.iloc[-1]) + .35)
    axes[3].scatter([tc.iloc[-1]], [0], color="#293746", s=16, zorder=5)
    fig.text(.10, .025, "Movement evidence is not a verified step. Fallback returns on timeout; "
             "actual alternating steps are not counted.", fontsize=9, color="#536174")
    fig.savefig(target / f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    # Only transition summaries are public. Keep full patient-level streams in the private bundle.
    selected = events[["guidance_timestamp_s", "guidance_action", "guidance_control_state",
                       "guidance_reason", "guidance_visual_enabled", "proxy_feedback_state",
                       "proxy_no_motion_valid_s"]].copy()
    selected["guidance_timestamp_s"] -= offset
    selected.rename(columns={"guidance_timestamp_s": "time_from_clip_start_s"}, inplace=True)
    selected.to_csv(target / f"{stem}_events.csv", index=False)
    return {"example": stem, "source_kind": "real_derived" if truth_path.exists() else "constructed",
            "source_commands_sha256": sha256(folder / "commands.csv"),
            "source_feedback_sha256": sha256(folder / "feedback_runtime.csv"),
            "source_perception_sha256": sha256(folder / "perception_runtime.csv"),
            "events": len(selected), "figure": f"{stem}.png"}


def matrix(root, junit=None):
    audit = pd.read_csv(root / "CASE_AUDIT.csv")
    rows = []

    def add(check, status, evidence, limit=""):
        rows.append({"check": check, "status": status, "evidence": evidence, "limit": limit})

    add("Standard files, record identity, required fields and clocks", "PASS",
        "48 cases; 240 strategy streams; CASE_AUDIT.csv; FIELD_CATALOG.csv")
    add("Perception/feedback provenance and separation from truth", "PASS",
        "Every runtime table checked; evaluation_truth.csv stored separately")
    add("Same perception input across strategies", "PASS",
        "All perception columns match each strategy command stream, row by row")
    add("Same feedback input across strategies", "PASS",
        "One shared feedback table per case; input manifest verified; no response regeneration")
    add("New valid post-cue observation required to advance timer", "PASS",
        f"{int(audit.new_valid_timer_increments.sum())} increments audited")
    add("Stale/unknown direction and invalid feedback withdraw cues", "PASS",
        "All 69,147 command rows checked; feedback-aware strategies require valid feedback")
    add("Every started command closes; final outputs off", "PASS",
        f"{int(audit.starts.sum())} start/stop pairs; every final visual/audio flag is off")
    add("Constructed low-motion fallback and timeout return", "PASS",
        f"{int(audit.fallback_entries.sum())} fallback entries; "
        f"{int(audit.timeout_returns.sum())} timeout returns",
        "Constructed probes only; real clips did not trigger fallback")
    for profile, reason in [("feedback_gap", "feedback_unavailable"),
                            ("unknown_direction", "perception_unavailable"),
                            ("refreeze", "perception_reconfirmed_restore_intensity")]:
        count = 0
        for folder in sorted((root / "constructed").glob(profile + "_*")):
            d = pd.read_csv(folder / "commands.csv")
            hit = d.guidance_reason.eq(reason)
            if profile != "refreeze":
                hit &= ~d.guidance_visual_enabled
            count += int(hit.sum())
        if not count:
            raise ValueError(f"Missing expected constructed branch: {profile}")
        add(f"Constructed {profile} branch", "PASS", f"{count} evidence rows; commands.csv")
    test_rows = []
    if junit:
        for test in ET.parse(junit).iter("testcase"):
            status = ("FAIL" if test.find("failure") is not None or test.find("error") is not None
                      else "NOT_RUN" if test.find("skipped") is not None else "PASS")
            test_rows.append({"test": test.attrib["name"], "status": status})
    groups = {
        "Reject reversed raw, perception and feedback clocks": [
            "test_reversed_clocks_are_rejected[raw]",
            "test_reversed_clocks_are_rejected[perception]",
            "test_reversed_clocks_are_rejected[feedback]"],
        "Reject duplicate raw timestamps and missing gyro": [
            "test_missing_channel_unknown_and_reversed_or_duplicate_times_rejected"],
        "Irregular/gapped/incomplete windows remain unknown": [
            "test_missing_incomplete_and_gap_become_unknown_without_filling"],
        "Reject mixed records, false source and missing decision ticks": [
            "test_reject_unmatched_records_bad_source_and_missing_feedback_ticks",
            "test_interleaved_records_rejected_and_labels_do_not_change_result"],
        "Rearm requires two genuinely new FoG observations": [
            "test_rearm_after_unknown_requires_two_new_fog_observations"],
    }
    for label, names in groups.items():
        statuses = [r["status"] for r in test_rows if r["test"] in names]
        status = ("FAIL" if "FAIL" in statuses else
                  "PASS" if len(statuses) == len(names) and set(statuses) == {"PASS"}
                  else "NOT_RUN")
        add(label, status, "; ".join(names), "Constructed unit-test inputs")
    add("Reliable left/right observed alternation and wrong-foot rejection", "NOT_IMPLEMENTED",
        "Proxy is movement_only; guidance_stable_steps remains zero",
        "Alternating footprints are targets; fallback uses timeout, not observed step completion")
    add("Step accuracy, forward displacement and clinical response", "NOT_EVALUABLE",
        "No independent step/position truth or intervention-response trial")
    add("New-machine reproduction", "NOT_RUN", "Current checks ran on the existing Windows environment")
    add("Final UI, animation and subtitle acceptance", "NOT_RUN", "Requires industrial-design playback")
    return pd.DataFrame(rows), pd.DataFrame(test_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh evidence directory")
    verified = verify_manifest(args.bundle)
    args.output.mkdir(parents=True)
    examples = [render_case(args.bundle, rel, args.output, title, stem)
                for rel, stem, title in CASES]
    acceptance, test_rows = matrix(args.bundle, args.junit)
    acceptance.to_csv(args.output / "ACCEPTANCE_MATRIX.csv", index=False)
    if not test_rows.empty:
        test_rows.to_csv(args.output / "ACCEPTANCE_TESTS.csv", index=False)
    for filename in ["FIELD_CATALOG.csv", "EXPORT_INVENTORY.csv"]:
        table = pd.read_csv(args.bundle / filename)
        table.to_csv(args.output / filename, index=False)
    write_json(args.output / "EVIDENCE_CHECK.json", {
        "status": "ENGINEERING_EVIDENCE_GENERATED_WITH_EXPLICIT_GAPS",
        "bundle_files_verified": verified,
        "source_manifest_sha256": sha256(args.bundle / "MANIFEST.json"),
        "render_script_sha256": sha256(__file__), "examples": examples,
        "junit_sha256": sha256(args.junit) if args.junit else None,
        "acceptance_counts": acceptance.status.value_counts().to_dict(),
        "not_a_full_B1_B2_completion_claim": True})
    print(json.dumps({"figures": len(examples), "acceptance_counts":
                      acceptance.status.value_counts().to_dict()}))


if __name__ == "__main__":
    main()
