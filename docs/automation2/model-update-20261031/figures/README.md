# English model-testing figures

| Figure | What it shows | Scope |
|---|---|---|
| [01](01_original_vs_generated_signal.png) | Original signal and one randomly perturbed version | Illustration, not a new participant |
| [02](02_generated_data_accuracy.png) | Original versus perturbed accuracy | 24 development clips, three variants per clip |
| [03](03_validation_accuracy_and_recall.png) | Accuracy and FoG recall | 920 common development-validation windows |
| [04](04_validation_errors.png) | False positives and false negatives | Window counts, not independent events |
| [05](05_noise_robustness.png) | FoG F1 under added noise | Engineering perturbations of the same development data |

Generated variants reuse source labels by assumption and were not independently re-annotated. These figures are not participant-held-out TEST results or treatment-effect evidence. Data and denominators are documented in ../02_Results/.
