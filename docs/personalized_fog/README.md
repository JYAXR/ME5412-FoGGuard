# Personalized FoG comparison

This optional study compares population training, personal training and target-person
threshold calibration using the **same bilateral RF detector**. The original three
delivered detectors, locked TEST results and A-group showcase remain the main delivery.

## Results and English figures

**[Open the chart/data report](presentation/index.html)** — three PNG/SVG figures,
five aggregate CSV tables, exact counts and concise explanations. Protocol and blocks
were committed before fitting (`7a608dc`, `4c75d33`); see [the pre-fit lock](PROTOCOL_LOCK.json).

The paired comparison includes 8 of 14 TRAIN participants, 31 records and 4,090
supported windows (1,298 FoG). Of 58 planned outer folds, 31 are eligible, 19 lack
enough positive adaptation windows and 8 have no supported evaluation windows.

| Strategy | Precision | Recall | FoG F1 | Non-FoG false-positive rate |
|---|---:|---:|---:|---:|
| Population | 0.659 | 0.723 | 0.689 | 17.4% |
| Personal | 0.631 | 0.881 | 0.735 | 24.0% |
| Population + calibration | 0.562 | 0.764 | 0.648 | 27.7% |

Personal training improves pooled F1 by 0.046, with higher false-positive rate and
lower precision. Four participants improve in F1 and four worsen. Person-macro F1
rises 0.474 to 0.526, but person-macro recall falls 0.658 to 0.621. Pooled recall
therefore does not describe improvement for every person. Threshold calibration
does not improve pooled F1; the predeclared protocol was not retuned after scoring.

Seven paired participants use within-session record holdout (3,992 windows); only
one uses session holdout (98 windows, 9 FoG), where personal F1 is lower. These
results do not establish across-day benefit or identify the cause of the original
errors. They are not directly comparable with the original four-person TEST scores.

Population-only controls with no positive supported windows remain in the report;
their FoG recall/F1 are not evaluable. Six participants cannot support the paired
personal study, so the paired cohort is not representative of everyone in TRAIN.

[Independent verification](presentation/INDEPENDENT_VERIFICATION.json) replays
12,270 saved-model predictions and independently checks metric counts, target
exclusion, blocked separation, adaptation-only threshold selection and 48 original
TEST/showcase artifact hashes. Public results are anonymous aggregates; raw signals,
weights and per-window evaluation files remain outside Git.

## Fixed protocol

The study uses only the original 14-person TRAIN pool. The original four development
participants and four TEST participants are not featurized, trained or scored. This
is exploratory blocked cross-validation, not a fresh blind test.

| Strategy | Classifier fit | Threshold | Target-person labeled data |
|---|---|---|---|
| Population | All other TRAIN-pool people; target excluded | Fixed 0.23 | None |
| Personal | Target-person adaptation blocks only | Fixed 0.23 | Same adaptation blocks available to calibration below |
| Population + calibration | Same population classifier | Chosen only on target adaptation blocks | Threshold calibration only |

All classifiers use the existing 66-feature bilateral gyro representation and
300-tree RF settings. The fixed threshold and RF settings are not optimized in
this study. Calibration searches the declared 0.05–0.95 grid, maximizing adaptation
F1, then balanced accuracy, then the higher threshold. Its benefit must be reported
as access to labeled personal calibration data, not zero-shot generalization.

Threshold 0.23 is inherited from prior TRAIN-based design. Target exclusion refers
to the current population classifier fit; this is not a newly isolated benchmark
with no prior knowledge of the TRAIN pool.

Personal RF fitting and population threshold calibration use the same labeled
adaptation data budget, in different ways. Personal thresholds remain fixed while
the calibrated strategy explicitly tunes its threshold. The comparison concerns
these complete strategies, not a controlled proof that participant variability
caused the original errors.

## Blocked evaluation and feasibility

Multiple-session participants use leave-one-session-out. One-session participants
use leave-one-record-out. Whole records define the split; windows are extracted
independently within each record and never cross adaptation/evaluation. Session IDs
are grouping keys, not independently verified dates, so the result is not a
prospective next-day trial. Both split kinds are reported separately.

Each adaptation fold needs at least 10 positive and 10 negative supported windows.
Eligibility depends on adaptation labels and input support, not held-out labels or
predictions. Unsupported evaluation blocks and insufficient adaptation support are
retained in the feasibility count. Overlapping windows are not independent samples.

All three strategies are compared on exactly the same eligible held-out windows.
Report pooled P/R/F1, false-positive rate, causal false-alarm duration, participant
macro results and per-person differences. If a participant has no positive
evaluation windows, FoG F1/recall are not evaluable; do not interpret an all-negative
accuracy as successful FoG detection. Population-only negative controls are kept
separately rather than hiding people who cannot support personal training.

## Run and preserve evidence

Use Python 3.12 and the existing project dependencies. Each plan needs a new directory.

```bash
python -X utf8 scripts/run_personalized_fog.py plan --output-dir reports/personalized_fog_NEW
python -X utf8 scripts/run_personalized_fog.py run --output-dir reports/personalized_fog_NEW
python -X utf8 scripts/verify_personalized_fog.py --run reports/personalized_fog_NEW --output reports/personalized_fog_NEW/INDEPENDENT_VERIFICATION.json
python -X utf8 scripts/build_personalized_fog_report.py --run reports/personalized_fog_NEW --output-dir reports/personalized_fog_presentation_NEW
```

The plan writes record/session assignments, supported window metadata, feature cache,
configuration and source/data/input hashes before fitting. Running requires those
hashes to match and refuses to replace an existing run. Runtime CSV files omit truth;
evaluation CSV files store it separately. Weights, raw identities, cached windows and
per-window CSV files stay outside Git. Calibrated-model JSON files reference saved
population weights to avoid duplicating a forest per fold.

The class `PersonalStudyModel.load(path)` loads a population/personal joblib file or
a calibration JSON and exposes sensor-only `predict_frame`. These research models
do not replace the main pipeline automatically.
