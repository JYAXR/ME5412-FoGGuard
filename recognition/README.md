# Real-data recognition audit

This tool verifies raw windows, participant-held-out predictions, recording
support and sample-level event metrics. Run it with a matching raw dataset,
OOF prediction file and training report. Missing predictions remain unknown.

The three selected A1 FoG models use their dedicated evaluator:
`scripts/evaluate_selected_fog.py`. Their FoG-only output is not a complete
input for this generic combination-prediction audit.

See [evaluation semantics](../docs/automation2/RECOGNITION.md),
[B2 tasks](../docs/tasks/B2.md), and the [project status](../docs/PROJECT_STATUS.json).

```bash
python -X utf8 -m pytest tests recognition -q
```

Do not commit raw data, model weights or generated runs. A model's final
predictions on its training data do not provide holdout evidence.
