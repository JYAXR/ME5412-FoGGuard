"""Isolated equal-person loss and TRAIN-only activity auxiliary supervision."""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from fogguard.modeling.a1_temporal_cnn import (
    TemporalClassifier,
    TemporalFoG,
    TemporalNetwork,
    history_indices,
)

ARMS = ("control", "person_balanced", "activity_aux")


def person_weights(groups, balanced):
    groups = np.asarray(groups, int)
    people, counts = np.unique(groups, return_counts=True)
    if not len(groups):
        raise ValueError("No fit participants")
    by_person = {int(p): len(groups) / (len(people) * int(n)) if balanced else 1.0
                 for p, n in zip(people, counts, strict=True)}
    weights = np.array([by_person[int(p)] for p in groups], dtype=np.float32)
    audit = [{"subject_id": int(p), "fit_windows": int(n), "window_weight": by_person[int(p)],
              "total_weight": by_person[int(p)] * int(n)} for p, n in zip(people, counts, strict=True)]
    return weights, audit


def activity_targets(windows):
    # Unlabeled / ambiguous activity is excluded from CE, not mapped to non-FoG.
    return np.array([w.activity_label - 1 if w.activity_label in range(1, 8) else -100
                     for w in windows], dtype=np.int64)


class ContextNetwork(TemporalNetwork):
    def __init__(self, dropout=0.4):
        super().__init__("cnn_gru", dropout)
        self.activity = nn.Linear(32, 7)
        nn.init.xavier_uniform_(self.activity.weight)
        nn.init.zeros_(self.activity.bias)

    def joint_forward(self, values, lengths):
        if values.ndim != 4 or values.shape[1:] != (3, 3, 120):
            raise ValueError("Expected N x 3 x 3 x 120 gyro history")
        if lengths.shape != (len(values),) or ((lengths < 1) | (lengths > 3)).any():
            raise ValueError("Invalid history lengths")
        valid = torch.arange(3).unsqueeze(0) < lengths.unsqueeze(1)
        encoded = values.new_zeros((len(values), 3, 24))
        encoded[valid] = self.encode(values[valid])
        packed = pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        h = hidden[-1]
        fog = self.base.output(self.base.drop(F.relu(self.base.dense(h)))).squeeze(-1)
        return fog, self.activity(h)

    def forward(self, values, lengths):
        # No activity target or hard gate appears in the inference graph.
        return self.joint_forward(values, lengths)[0]


def training_objective(fog, activity, y, targets, weights, auxiliary_weight):
    fog_loss = (F.binary_cross_entropy_with_logits(fog, y, reduction="none") * weights).mean()
    valid = targets.ne(-100)
    auxiliary = F.cross_entropy(activity[valid], targets[valid]) if valid.any() else activity.sum() * 0
    return fog_loss + auxiliary_weight * auxiliary, fog_loss, auxiliary


class ContextClassifier(TemporalClassifier):
    def __init__(self, kind, config, seed, objective):
        if kind not in ARMS:
            raise ValueError("Unknown context-learning arm")
        super().__init__(kind, config, seed)
        self.objective = dict(objective)

    def fit(self, values, lengths, labels, *, groups, activities, validation=None, fixed_epochs=None):
        if (validation is None) == (fixed_epochs is None):
            raise ValueError("Choose checkpoint validation OR fixed epochs")
        if fixed_epochs is not None and not 1 <= fixed_epochs <= self.config["max_epochs"]:
            raise ValueError("Invalid fixed epochs")
        torch.set_num_threads(self.config["cpu_threads"])
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        x, l = torch.as_tensor(values, dtype=torch.float32), torch.as_tensor(lengths, dtype=torch.long)
        y, a = torch.as_tensor(labels, dtype=torch.float32), torch.as_tensor(activities, dtype=torch.long)
        if set(y.tolist()) != {0.0, 1.0} or not torch.isfinite(x).all() or len(groups) != len(y) or len(a) != len(y):
            raise ValueError("Fit requires finite aligned inputs, targets, people and both classes")
        if not set(a.tolist()) <= {-100, *range(7)}:
            raise ValueError("Unexpected activity targets")
        self.sample_weights_, self.weight_audit_ = person_weights(groups, self.objective["person_balance"])
        weights = torch.from_numpy(self.sample_weights_)
        self.fit_people_ = tuple(sorted(map(int, set(groups))))
        self.activity_target_counts_ = {str(v): int((np.asarray(activities) == v).sum()) for v in (-100, *range(7))}
        if validation is not None:
            vx, vl, vy = (torch.as_tensor(validation[0], dtype=torch.float32),
                          torch.as_tensor(validation[1], dtype=torch.long),
                          torch.as_tensor(validation[2], dtype=torch.float32))
            if set(vy.tolist()) != {0.0, 1.0}:
                raise ValueError("Checkpoint requires both FoG classes")
        self.network_ = ContextNetwork(self.config["dropout"])
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.config["learning_rate"],
                                      weight_decay=self.config["weight_decay"])
        generator = torch.Generator().manual_seed(self.seed)
        best, wait, state = float("inf"), 0, None
        self.history_ = []
        limit = self.config["max_epochs"] if validation is not None else fixed_epochs
        for epoch in range(1, limit + 1):
            self.network_.train()
            totals = np.zeros(3)
            order = torch.randperm(len(y), generator=generator)
            for start in range(0, len(y), self.config["batch_size"]):
                index = order[start:start + self.config["batch_size"]]
                optimizer.zero_grad()
                fog, activity = self.network_.joint_forward(x[index], l[index])
                objective, fog_loss, auxiliary = training_objective(fog, activity, y[index], a[index], weights[index], self.objective["activity_weight"])
                loss = objective + self.config["kernel_l2"] * self.network_.kernel_penalty()
                loss.backward()
                optimizer.step()
                totals += np.array([float(loss.detach()), float(fog_loss.detach()), float(auxiliary.detach())]) * len(index)
            self.network_.eval()
            row = dict(zip(("train_loss", "train_fog_objective", "train_aux_batch_mean_ce"), map(float, totals / len(y)), strict=True))
            row["epoch"] = epoch
            if validation is not None:
                with torch.inference_mode():
                    total = 0.0
                    for start in range(0, len(vy), self.config["batch_size"]):
                        stop = start + self.config["batch_size"]
                        # Same unweighted FoG-only checkpoint objective in every arm.
                        total += float(F.binary_cross_entropy_with_logits(self.network_(vx[start:stop], vl[start:stop]), vy[start:stop])) * len(vy[start:stop])
                    val_loss = total / len(vy) + self.config["kernel_l2"] * float(self.network_.kernel_penalty())
                if not np.isfinite(val_loss):
                    raise ValueError("Nonfinite checkpoint loss; stop without retry")
                row["checkpoint_loss"] = val_loss
                if val_loss < best - self.config["min_delta"]:
                    best, wait, state = val_loss, 0, copy.deepcopy(self.network_.state_dict())
                    self.best_epoch_ = epoch
                else:
                    wait += 1
            self.history_.append(row)
            if validation is not None or epoch == 1 or epoch % 5 == 0 or epoch == limit:
                print(f"{self.kind} epoch {epoch:02d}/{limit}: {row}", flush=True)
            if validation is not None and wait >= self.config["patience"]:
                break
        if validation is not None:
            self.network_.load_state_dict(state)
        else:
            self.best_epoch_ = fixed_epochs
        self.network_.eval()
        self.classes_ = np.array([0, 1])
        self.parameter_count_ = sum(p.numel() for p in self.network_.parameters())
        return self


class ContextFoG(TemporalFoG):
    def predict_windows(self, windows):
        result = super().predict_windows(windows)
        result["model_version"] = f"a1_context_learning_v1/{self.variant}"
        result["context_windows"] = (history_indices(windows) >= 0).sum(axis=1)
        return result
