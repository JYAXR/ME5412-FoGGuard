# Automation 2 handoff for coding agents

## Read order and current scope

1. Repository `AGENTS.md` and `AUTOMATION2_START_HERE.md`.
2. `docs/automation2/status.json` for machine-readable state, metrics and entry points.
3. `docs/COLLABORATION.md`, the relevant role handoff, then source and tests.
4. `docs/automation2/evidence/2026-10-23/` for immutable evaluated snapshots.

This is an additive contribution based on upstream PR #24 SHA
`58135167a3be0aaa4c93e9c138d732efda9dd38a`, which contains PR #23. It is not a claim
that either PR is merged. The focused PR base is `codex/automation1-guidance`.
Do not copy this fork over upstream main or overwrite another teammate's branch.

## What code belongs to which layer

| Layer | Entry | Evidence / limits |
|---|---|---|
| Original robot/model/context/controller | Existing `src/fogguard/{data,modeling,context,cueing}/` | Preserved baseline; original controller still default |
| Automation 2 initial audit | `scripts/run_automation2_evaluation.py`, `src/fogguard/evaluation/automation2*.py` | 112 offline scenario/strategy pairs; 21 PASS + 1 known limitation; raw-window onset and cue-start semantics documented in original audit handoff |
| Optional marching simulation | `src/fogguard/cueing/march.py`, `src/fogguard/complete.py` | 21 built-in cases + 1 supplied robot2 handoff = 22 × 5 = 110 runs; constructed candidates and simulated responses |
| Full simulation viewer | `scripts/run_complete_demo.py`, `src/fogguard/presentation/` | Reads saved commands; real sound is user opt-in; configured coordinates are not measured foot poses |
| Raw real-data recognition audit | `recognition/evaluate.py`, `recognition/diagnostics.py` | Saved participant-held-out OOF, original raw labels, all 106 raw records |
| Check runner | `scripts/run_automation2_delivery_checks.py` | Root invocation; includes both test suites, Ruff, baseline audit and demo; optional raw audit |

`replay_guidance` gained an optional `controller_type` parameter, defaulting to the original
`GuidanceController`. `MarchGuidanceController` subclasses that controller. Do not interpret
the new extension as a replacement of the team's shared types or an approved online API.

## Data and causal semantics

- FoG-STAR raw MD5: `952a37ab147da35e6d4e7a1e9bac44cb`; DOI: `10.5281/zenodo.17838806`.
- Use the saved OOF and matching training report from the recognition release asset, or
  regenerate using the original subject-group training script. Do not use the final model's
  predictions on its training set to claim generalization.
- Window audit reconstructs every original 120-sample window, checks keys/end time/truth
  fraction and all folds/thresholds. This is a structural/provenance audit, not independent
  retraining or proof of the complete historical training process.
- Raw `timestamp` increments are ~1/60 and original code uses seconds, despite the dataset
  README's ms label. Preserve the source; record the discrepancy.
- Long IMU gaps are not filled. All 106 raw records are shown; only 95 contain valid OOF
  windows. Missing predictions are never labeled as model-negative observations.
- Window FoG truth: >=50% of 120 samples. Manifestation group: strict majority among
  **frozen samples only**. This is truth-stratified binary evaluation, not a subtype classifier.
- Raw event truth uses continuous sample labels and includes events without model support.
  Decisions begin at window end and hold at most 0.5s; no backdating into input windows.
  Coverage is any positive overlap, not one-to-one event matching or event F1. Also report
  duration precision/recall. Conditional onset delay excludes completely missed events;
  preexisting alarms are listed separately. These are offline decisions, not hardware cue latency.
- Participant cluster bootstrap resamples people, not correlated windows; it does not retrain.
- Synthetic or constructed inputs and preset responses validate engineering behavior only.
  Do not compare raw-event coverage (101 events), integrated window events (81 in older
  results), marching cases or clinical benefit as though they were the same endpoint.

## Verification and interpretation

From the repository root with Python 3.12 and dependencies installed:

```bash
python -X utf8 scripts/run_automation2_delivery_checks.py
```

Use `pip install -r requirements.txt -c docs/automation2/tested-constraints.txt` to match
the tested direct dependency versions without editing the team's original requirements.

For a full raw-data rerun, add `--raw`, `--oof`, `--training-report` together. Without them,
the runner explicitly records `real_data_audit: NOT_RUN`; dated real-data snapshots are
available but were not re-evaluated by that run. Run directories are fresh and nonempty
explicit targets are refused. Subprocesses use UTF-8 and bounded math-library threads.

Previous evaluated source: 241 tests passed (203 cumulative project + 38 recognition),
Ruff passed, and an extracted package regenerated identical recognition artifacts.
Re-run on changed code; historical evidence is not a check for a new commit. CI will repeat
the offline checks on Windows and Ubuntu, but does not download/train real data.

## Known limits and teammate boundaries

- FoG precision ~56%, recall ~66%; this is an imperfect engineering baseline.
- 11 raw records have missing required channels and no valid windows; 18 of 25 fully
  uncovered true events have no model support. Inspect `missing-input-records.csv`.
- Stationary/voluntary-stop false prompts remain a known limitation, not a hidden PASS.
- Source changes in this contribution preserve default behavior; shared online contracts
  still need the robot/context and controller owners' review.
- Final design/presentation remains a team deliverable; editable deck/video are drafts.
- No clinical efficacy, real foot tracking or validated patient response model is provided.

See `CONTINUING_AND_UPLOAD.md` before a subsequent upload. Keep dated evidence immutable,
make a new snapshot for changed experiments, and record which code/config/inputs produced it.
