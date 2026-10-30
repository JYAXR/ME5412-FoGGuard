"""Generate sensor variants from real inputs for bounded robustness experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_variant(sensor: pd.DataFrame, seed: int):
    """Apply fixed mild random gain, offset and noise; preserve identity and missingness.

    These perturbations are not a generative patient model, and transferred labels
    have not been independently reviewed for each generated version.
    """
    if {"fog", "activity", "fog_severity"} & set(sensor.columns) or any(
        str(c).startswith("truth_") for c in sensor.columns
    ):
        raise ValueError("Generator must not receive labels")
    out = sensor.copy(deep=True)
    rng = np.random.default_rng(seed)
    parameters = {}
    for column in sensor.columns:
        if "_gyro_" not in column and "_acc_" not in column:
            continue
        source = pd.to_numeric(sensor[column], errors="raise").to_numpy(dtype=float)
        if np.isinf(source).any():
            raise ValueError("Infinite sensor values are invalid")
        gyro = "_gyro_" in column
        gain = float(rng.uniform(0.97, 1.03))
        offset = float(rng.uniform(-0.3, 0.3) if gyro else rng.uniform(-0.003, 0.003))
        sigma = 0.5 if gyro else 0.005
        out[column] = source * gain + offset + rng.normal(0.0, sigma, len(source))
        parameters[column] = {"gain": gain, "offset": offset, "noise_std": sigma}
    return out, parameters
