# Evaluation and integration development

Read the repository `AGENTS.md`, project README, `docs/PROJECT_STATUS.json`,
`docs/tasks/B2.md` and `docs/INTERFACE_CONTRACT.md`. Current roles are A1
freezing, A2 direction/perception fusion, B1 feedback/control, B2 evaluation,
and industrial design for the final presentation.

Run `python -X utf8 scripts/run_automation2_delivery_checks.py` from the root.
It covers `tests` and `recognition`, Ruff, scenario simulation and paired
strategy auditing. Without all raw/OOF/training-report inputs, real-data
auditing is NOT_RUN. Use fresh directories and preserve input/config hashes.

The demo uses constructed candidates and simulated responses. The selected
A1 models have a dedicated evaluator; generic compatibility audits do not
provide their scores. New direction and motion feedback adapters must be
implemented and verified before claiming a complete real-data pipeline.

Shared types and local cueing contracts remain separate. Truth cannot drive
runtime decisions, unknown feedback cannot count as no step, and a movement
proxy cannot be relabeled as measured forward displacement. Dated audit
manifests document only their specific inputs and scope.

Use `docs/COLLABORATION.md` for publishing and receiver review.
