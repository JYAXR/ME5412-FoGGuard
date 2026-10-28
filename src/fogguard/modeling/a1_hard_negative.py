"""Training-variant metadata around the existing sensor-only RF inference."""

from dataclasses import dataclass

from fogguard.modeling.a1_multimodel import MultiModelFoG


@dataclass
class HardNegativeFoG(MultiModelFoG):
    training_variant: str = "hard_negative"

    def predict_windows(self, windows):
        result = super().predict_windows(windows)
        result["model_version"] = f"a1_hard_negative_v1/{self.training_variant}"
        return result
