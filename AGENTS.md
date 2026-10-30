# Working in FoGGuard

Read `README.md`, `docs/PROJECT_OVERVIEW.md`, `docs/ROLES_AND_ROADMAP.md`,
`docs/PROJECT_STATUS.json` and the relevant `docs/tasks/` task before editing.
Use `docs/ARCHITECTURE.md` and `docs/INTERFACE_CONTRACT.md` for module boundaries.

- A1's completed algorithm delivery consists of exactly three FoG methods:
  `ankle_history_rf`, `bilateral_rf`, `ankle_cnn`. Entry points are
  `predict_selected_fog.py`, `train_selected_fog.py`, `evaluate_selected_fog.py`
  and `bundle_selected_fog.py`. A2 owns direction and perception fusion; B1
  owns motion feedback and cueing; B2 owns evaluation and joint integration.
- A1/A2 perception, locked TEST and the unified chart/data showcase are delivered
  on main. Current A-group tasks are maintenance and concrete handoff support;
  real feedback/control belongs to B1, receiver/joint acceptance to B2, and
  final PPT/visualization/video to industrial design. Keep this distinction in
  current task/status documents; known model limits are not unfinished training.
- Project-facing documentation describes the current design, implementation,
  remaining work and evidence. Keep a single consistent role plan and avoid
  experiment diaries, superseded task suggestions and cleanup narratives.
- Preserve unrelated teammate code and uncommitted work. Inspect Git status
  and the contribution branch first. Do not reset, clean, force-push, merge
  or overwrite another person's work outside the user request.
- `src/fogguard/types.py` and `interfaces.py` are shared contracts. Follow
  `docs/COLLABORATION.md` for interface review. New adapters are implementation
  tasks, not already-approved shared APIs.
- Source availability is explicit. Completed A1 windows become usable at
  window end; perception adapters hold each result for at most 0.5 seconds.
  No future-filled labels, unbounded stale predictions or cross-record state.
- Labels belong in evaluation, never runtime decisions. Missing sensor or
  feedback data is unknown, not a normal observation or evidence of no step.
- FoG classification is binary. Truth-stratified manifestation metrics do
  not mean the model predicts subtypes. Motion proxy evidence is not measured
  forward displacement; simulated response evidence is not clinical efficacy.
- The engineering demo uses constructed candidates and simulated responses.
  Do not describe it as the completed four-module real-data pipeline. The
  legacy A1 tables remain development validation. The separate locked report in
  `docs/a_group/` is participant-held-out TEST, not a clinical score. TEST is now
  consumed for selection: never retune on it and claim a fresh independent test.
- Do not commit raw datasets, model weights, credentials, clinical metadata,
  local absolute paths or generated run directories. Stage explicit paths.
- Preserve dated verification evidence and release assets. Use fresh output
  directories and input/config/code hashes for new runs. Current project
  status is `docs/PROJECT_STATUS.json`; check manifests describe their own
  bounded evaluation scope, not whole-project completion.
- Run checks from the repository root using Python 3.12:
  `python -X utf8 scripts/run_automation2_delivery_checks.py`.
  This includes `tests` and `recognition`, Ruff, demos and strategy audits.
  Without raw/OOF/report inputs, real-data auditing is NOT_RUN, not PASS.
- Work on contribution branches and require teammate review before merging.
  Use `docs/COLLABORATION.md` for handoff and publication.
