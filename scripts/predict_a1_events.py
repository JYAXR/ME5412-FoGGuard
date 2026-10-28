"""FoG-only scores plus optional causal event gate; never read runtime labels."""

import argparse
import json
from pathlib import Path

from fogguard.data.fogstar import load_fogstar_frame
from fogguard.modeling.a1_event import EventPolicy, apply_event_policy
from fogguard.modeling.a1_fog import A1FogModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    model = A1FogModel.load(args.model)
    saved = json.loads(args.policy.read_text("utf-8"))
    policy = EventPolicy(saved["threshold"], saved["confirmation_windows"])
    if abs(model.threshold - policy.threshold) > 1e-12:
        raise ValueError("Model and event policy thresholds differ")
    positions = tuple(dict.fromkeys(c.split("_", 1)[0] for c in model.channel_names))
    frame = load_fogstar_frame(args.input, sensor_positions=positions, require_labels=False)
    result = apply_event_policy(model.predict_frame(frame), policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} sensor-only score/event rows")


if __name__ == "__main__":
    main()
