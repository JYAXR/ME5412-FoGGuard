# Real-data recognition audit

This folder is an independent evaluation tool. It does not replace the existing model.
Read [the contribution start page](../AUTOMATION2_START_HERE.md),
[the full recognition report](../docs/automation2/RECOGNITION.md) and
[AI handoff](../docs/automation2/AI_HANDOFF.md).

From the repository root after `pip install -e .`:

```bash
python -X utf8 recognition/evaluate.py --raw data/raw/fogstar/sensor_data.csv --predictions path/to/fogstar_oof_predictions.csv --training-report path/to/fogstar_training.json --output reports/recognition-new-run
python -X utf8 -m pytest tests recognition -q
```

Raw data and model files are excluded from Git. The dated recognition Release asset contains
the saved OOF inputs and a self-contained viewer; Git tracks only aggregate, deidentified evidence.
Use participant-held-out OOF predictions; the deployed all-subject model is not a holdout test.
