# Five roles and three-week roadmap

## Role responsibilities

| Role | Primary responsibility | Concrete outputs | Backup/reviewer |
|---|---|---|---|
| Robotics 1 — Data & FoG | canonical data, replay, FoG baseline | loader, windowing, FoG score, dataset notes | Automation 2 |
| Robotics 2 — Turning & integration | turning intent and pipeline integration | turn baseline, orchestrated replay, integration fixes | Automation 1 |
| Automation 1 — Cueing & safety | adaptive control logic | state machine, refractory logic, fail-safe tests | Robotics 2 |
| Automation 2 — Evaluation & QA | experiments and reproducibility | metrics, labelled scenarios, regression tests, final tables | Robotics 1 |
| Industrial Design — UX & storytelling | safe understandable demo | Streamlit flow, cue visualization, usability test, presentation assets | whole team |

Every role also reviews one other role's pull request. The project coordinator can be any member; coordination does not replace a technical role.

## Week 1 — contracts and vertical slice

Goal: one synthetic file passes through the complete pipeline.

- Freeze `SensorWindow`, result, context, and cue contracts.
- Generate canonical synthetic data and validate the loader.
- Implement explainable FoG and turn baselines.
- Implement the cueing policy and a one-page Streamlit shell.
- Establish metrics, tests, branch rules, and the task board.

Exit check: a fresh clone can generate data, run tests, and show one end-to-end replay.

## Week 2 — evidence and user experience

Goal: credible behavior on representative scenarios.

- Add at least three scenarios: normal walking, straight-walk FoG, turning FoG.
- Tune thresholds only on a documented development split.
- Measure precision, recall, F1, turn accuracy, false cues/minute, and cue latency.
- Improve controller state/refractory behavior and surface its reason in the UI.
- Conduct a five-person heuristic/usability walkthrough and record changes.

Exit check: results are reproducible and the team can explain one failure case per module.

## Week 3 — freeze, validate, present

Goal: a reliable five-minute demonstration and honest engineering report.

- Freeze interfaces and features by midweek.
- Run the final evaluation and archive config/results.
- Add failure-safe, missing-data, and boundary tests.
- Rehearse demo recovery using the synthetic dataset.
- Finish architecture figure, limitations, ethics/safety statement, slides, and demo video backup.

Exit check: tagged demo build, final metrics table, presentation, and clear limitations.
