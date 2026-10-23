# Working in FoGGuard

Read `AUTOMATION2_START_HERE.md`, `docs/automation2/AI_HANDOFF.md` and
`docs/automation2/status.json` before continuing the Automation 2 contribution.
For the whole project, also read `docs/COLLABORATION.md` and the relevant role's handoff.

- Preserve unrelated teammate code, files, commits and uncommitted work. Inspect Git status
  and the target branch first. Use new commits on a contribution branch; do not reset,
  clean, force-push, merge or replace another person's work outside the current user request.
- PR #23 and #24 are dependencies, not evidence of a merge into `main`. Verify their live
  state when integrating. This contribution starts at #24 commit
  `58135167a3be0aaa4c93e9c138d732efda9dd38a` and targets its head branch for a focused diff.
- Existing role handoffs are historical module snapshots. The Automation 2 start page and
  status manifest describe the cumulative contribution; inspect code and rerun commands
  when resolving a discrepancy. Do not silently rewrite teammate documentation.
- Shared `src/fogguard/types.py` and `interfaces.py` follow the team's interface review
  process in `docs/ARCHITECTURE.md`. Marching is an optional simulation extension, not an
  approved shared online contract. The original controller remains the default.
- Distinguish synthetic IMU, constructed candidates, simulated responses, saved OOF model
  predictions, window labels, sample-level events and downstream cueing events. Never
  present simulation outcomes as clinical efficacy or reuse training-set predictions as OOF.
- The detector is binary FoG/non-FoG. Manifestation groups use **truth annotations** for
  evaluation; the model does not classify shuffling/trembling/akinesia.
- Include all 106 raw records in event coverage, even though only 95 have valid predictions.
  Missing inputs are not a confident non-FoG result. Do not fill long missing signals with zeros.
- Do not commit raw datasets, deployed models, credentials, clinical metadata, local absolute
  paths or generated run directories. Commit explicit source/docs paths, not `git add .`
  after running data workflows. Aggregated dated evidence is under `docs/automation2/evidence/`.
- Run checks from the repository root with Python 3.12:
  `python -X utf8 scripts/run_automation2_delivery_checks.py`.
  This covers `tests` **and** `recognition`; plain `pytest` follows the original `testpaths`
  and does not include the separate recognition suite. With no raw/OOF inputs, real-data
  reproduction is explicitly NOT_RUN, not PASS.
- Preserve dated evidence snapshots and release assets. For a new evaluation, use a new run
  directory, save input/config/code hashes and update the status manifest with its scope.
  Follow `docs/automation2/CONTINUING_AND_UPLOAD.md` for subsequent GitHub updates.
