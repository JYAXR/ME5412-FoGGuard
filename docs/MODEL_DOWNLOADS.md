# Pretrained model download

**[Download the four-model ZIP](https://github.com/JYAXR/ME5412-FoGGuard/releases/download/a-group-models-2026-10-30/FoGGuard-A-Group-Models-20261030.zip)**
(25.1 MB) · **[Release page and checksums](https://github.com/JYAXR/ME5412-FoGGuard/releases/tag/a-group-models-2026-10-30)**

The pretrained weights are GitHub Release assets. They are downloaded separately
from the source checkout; `git clone` does not download release assets.
**No retraining is needed.** These are the official A-group models used by the
locked TEST report, with the original weights and thresholds.

| Model | Extracted path under `artifacts/a-group-models/` |
|---|---|
| A1 left-ankle history RF | `fog-selected/ankle_history_rf/model.joblib` |
| A1 bilateral RF (fusion reference) | `fog-selected/bilateral_rf/model.joblib` |
| A1 left-ankle CNN | `fog-selected/ankle_cnn/model.joblib` |
| A2 current-direction RF | `a2-current/current/model.joblib` |

The optional personal-model study is separate and is not included in this release.
Raw patient signals and per-window patient outputs are not distributed in the ZIP.
Configuration, tested dependency versions, checksums, an integrity checker and
synthetic-only input/output examples are included.

## Install, verify and extract

Use Python 3.12 in a clone of `main`. Matching source snapshot:
[`b04ea861`](https://github.com/JYAXR/ME5412-FoGGuard/tree/b04ea8618e155d62891ca2970fc7c028ce3b1f06).
Save the ZIP in the repository root. Its SHA-256 is:

```text
b3a6a69c2c9860a2dd992f1dd0169e39514f7fa4f44a6272a7fc47be1a9770bf
```

On Windows, `Get-FileHash FoGGuard-A-Group-Models-20261030.zip -Algorithm SHA256`
prints the archive hash. Then run:

```bash
python -m pip install -r requirements-a2.txt -c docs/automation2/tested-constraints.txt
python -m pip install -r requirements-a1-deep.txt
python -m pip install -e .
python -m zipfile -e FoGGuard-A-Group-Models-20261030.zip artifacts/a-group-models
python artifacts/a-group-models/verify_models.py
```

Use a new extraction directory. CPU PyTorch is needed for `ankle_cnn`; RF
inference does not require it. Exact versions are recorded in the package's
`TESTED_VERSIONS.json`. The installed project supplies the Python classes needed
to load the joblib files. Use the trusted release and verify its hashes before loading.

## Run without the original dataset

The included example uses generated signals for interface checking:

```bash
python -X utf8 scripts/predict_selected_fog.py --bundle artifacts/a-group-models/fog-selected --method bilateral_rf --input artifacts/a-group-models/examples/synthetic_imu.csv --output reports/released_fog_NEW.csv
python -X utf8 scripts/predict_a2_turning.py --model artifacts/a-group-models/a2-current/current/model.joblib --input artifacts/a-group-models/examples/synthetic_imu.csv --output reports/released_direction_NEW.csv
python -X utf8 scripts/fuse_perception.py --fog reports/released_fog_NEW.csv --direction reports/released_direction_NEW.csv --output-dir reports/released_perception_NEW
```

Use `--method ankle_history_rf` or `--method ankle_cnn` to run the other A1
detectors. Use fresh output paths. The package contains expected synthetic
outputs for comparison; they are not accuracy or clinical-effectiveness evidence.

## Real-data use and integration

Replace the example input with a CSV satisfying the channel, unit, timestamp and
record-key requirements in [A1 model usage](A1_A2_START_HERE.md) and
[the data schema](DATA_SCHEMA.md). The original FoG-STAR dataset is obtained separately.
For the fixed development-only A1+A2 replay:

```bash
python -X utf8 scripts/validate_perception.py --raw data/raw/fogstar/sensor_data.csv --fog-bundle artifacts/a-group-models/fog-selected --method bilateral_rf --direction-run artifacts/a-group-models/a2-current --output-dir reports/released_development_NEW
```

The `a2-current/results.json` file preserves training provenance; its
`test_scored=false` describes the training run. The separate locked TEST was
subsequently evaluated and is documented in the [formal report](a_group/README.md).
Model performance, false alarms and direction-fusion coverage limits remain
unchanged. A2 predicts current direction, not reliable advance intention.
Perception output does not supply real motion feedback or cue control.

Release provenance and hashes: [release manifest](models/RELEASE_MANIFEST.json).
Full receiver guidance: [A1+A2 perception](a2/fusion/README.md).
