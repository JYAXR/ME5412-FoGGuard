# ME5412 FoGGuard

FoGGuard is a three-week ME5412 team prototype for replaying wearable gait data, detecting freezing of gait (FoG), recognizing turning intent, and selecting an adaptive cue. The final deliverable is an offline, reproducible Streamlit demonstration with quantitative evaluation—not a medical device or a real-time clinical system.

## Three-week MVP

The demo follows one stable pipeline:

```text
CSV / synthetic replay
        ↓ SensorWindow
FoG detector ────────┐
                     ├─ ControllerContext → adaptive cue → Streamlit timeline
turn-intent model ───┘
        ↓
offline metrics and scenario report
```

Success means the team can run one command, replay at least three scripted scenarios, explain every module boundary, and report FoG precision/recall/F1 plus turn-intent accuracy and cue latency.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
streamlit run src/fogguard/ui/app.py
```

Generate a small demo dataset with `python scripts/generate_demo_data.py`. Run checks with `pytest` and `ruff check .`.

## Repository map

| Path | Owner role | Purpose |
|---|---|---|
| `src/fogguard/data/` | Robotics 1 | CSV loading, validation, fixed-window replay |
| `src/fogguard/detection/` | Robotics 1 | FoG scoring and threshold baseline |
| `src/fogguard/turning/` | Robotics 2 | turning-intent recognition |
| `src/fogguard/cueing/` | Automation 1 | adaptive cueing state machine |
| `src/fogguard/ui/` | Industrial Design | Streamlit demo and interaction flow |
| `src/fogguard/evaluation/` | Automation 2 | metrics, experiments, reproducibility |
| `src/fogguard/types.py` | All | shared immutable data contracts |
| `src/fogguard/interfaces.py` | All | module protocols; change only by team agreement |
| `docs/` | All | architecture, data schema, roles, and workflow |

## Module contract

All modules exchange the dataclasses in `fogguard.types`:

- `SensorWindow`: one time-ordered IMU window in SI units.
- `DetectionResult`: FoG probability, Boolean prediction, and optional evidence.
- `TurnIntent`: left/right/straight class and confidence.
- `ControllerContext`: current predictions plus timing and refractory state.
- `CueCommand`: cue type, intensity, duration, and human-readable reason.

Implementations satisfy the protocols in `fogguard.interfaces`. Feature branches may replace a baseline without changing downstream code. Any interface change needs a short issue, agreement from affected owners, and an update to `docs/ARCHITECTURE.md`.

## Team workflow

1. Pick one GitHub Issue and create a short branch such as `feat/fog-baseline`.
2. Keep pull requests small; link the issue with `Closes #<number>`.
3. Require one teammate review before merging to `main`.
4. Never commit participant-identifiable data. Put local datasets under `data/raw/`, which is ignored.
5. At the end of each day, update the issue checklist and record blockers.

See [roles and roadmap](docs/ROLES_AND_ROADMAP.md), [collaboration rules](docs/COLLABORATION.md), and [architecture](docs/ARCHITECTURE.md).

## Safety and scope

This repository is for coursework and engineering evaluation. It does not diagnose, prevent, or treat disease. The cueing controller must default to `CueType.NONE` when inputs are missing, stale, or low confidence. Any human-subject data collection requires instructor and ethics approval outside this repository.
