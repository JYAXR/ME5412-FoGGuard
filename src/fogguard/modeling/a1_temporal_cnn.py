"""Bounded causal gyro history and a controlled CNN / single-direction GRU pair."""

import copy
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from fogguard.modeling.a1_author_cnn import GYRO_CHANNELS, AuthorFoGCNN, centered_gyro, gyro_windows
from fogguard.modeling.a1_fixed_split import prediction_frame
from fogguard.modeling.a1_fog import A1FogModel

LAGS = (8, 4, 0)


def history_indices(windows):
    """Oldest to newest complete windows; unused suffix slots are -1."""
    links = np.full((len(windows), 3), -1, dtype=int)
    run_start = 0
    for i, window in enumerate(windows):
        if i:
            previous = windows[i - 1]
            same_record = (window.subject_id, window.session_id, window.task_id) == (
                previous.subject_id, previous.session_id, previous.task_id)
            delta = window.start_s - previous.start_s
            if same_record and delta <= 0:
                raise ValueError("Windows must be chronological, without duplicates")
            if not same_record or not np.isclose(delta, 0.5, atol=1e-6, rtol=0):
                run_start = i
        chosen = [i - lag for lag in LAGS if i - lag >= run_start]
        links[i, :len(chosen)] = chosen
    return links


def sequence_batch(windows):
    current = centered_gyro(windows)
    links = history_indices(windows)
    lengths = (links >= 0).sum(axis=1)
    # Storage padding only; the encoder and GRU explicitly exclude invalid slots.
    values = np.zeros((len(windows), 3, 3, 120), dtype=np.float32)
    valid = links >= 0
    values[valid] = current[links[valid]]
    return values, lengths, links


class TemporalNetwork(nn.Module):
    def __init__(self, kind, dropout=0.4):
        super().__init__()
        if kind not in ("cnn_control", "cnn_gru"):
            raise ValueError("Unknown temporal architecture")
        self.kind = kind
        self.base = AuthorFoGCNN(dropout)
        if kind == "cnn_gru":
            self.gru = nn.GRU(24, 32, num_layers=1, batch_first=True, bidirectional=False)
            self.base.dense = nn.Linear(32, 48)
            nn.init.xavier_normal_(self.base.dense.weight)
            nn.init.zeros_(self.base.dense.bias)

    def encode(self, values):
        values = self.base.conv2(self.base.conv1(values))
        values = self.base.drop(F.max_pool1d(values, 4))
        values = self.base.drop(F.max_pool1d(self.base.conv3(values), 3))
        return self.base.conv4(values).mean(dim=2)

    def forward(self, values, lengths):
        if values.ndim != 4 or values.shape[1:] != (3, 3, 120):
            raise ValueError("Expected N x 3 history slots x 3 gyro x 120 samples")
        if lengths.shape != (len(values),) or ((lengths < 1) | (lengths > 3)).any():
            raise ValueError("Invalid packed history lengths")
        if self.kind == "cnn_control":
            return self.base(values[torch.arange(len(values)), lengths - 1])
        valid = torch.arange(3).unsqueeze(0) < lengths.unsqueeze(1)
        encoded = values.new_zeros((len(values), 3, 24))
        encoded[valid] = self.encode(values[valid])
        packed = pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.base.output(self.base.drop(F.relu(self.base.dense(hidden[-1])))).squeeze(-1)

    def kernel_penalty(self):
        return self.base.kernel_penalty()


class TemporalClassifier:
    def __init__(self, kind, config, seed):
        self.kind, self.config, self.seed = kind, dict(config), seed

    def _loss(self, x, lengths, y):
        return F.binary_cross_entropy_with_logits(self.network_(x, lengths), y)

    def fit(self, values, lengths, labels, *, validation=None, fixed_epochs=None):
        if (validation is None) == (fixed_epochs is None):
            raise ValueError("Choose checkpoint validation OR fixed epoch fit")
        if fixed_epochs is not None and not 1 <= fixed_epochs <= self.config["max_epochs"]:
            raise ValueError("Invalid fixed epoch count")
        torch.set_num_threads(self.config["cpu_threads"])
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        x = torch.as_tensor(values, dtype=torch.float32)
        lengths = torch.as_tensor(lengths, dtype=torch.long)
        y = torch.as_tensor(labels, dtype=torch.float32)
        if set(y.tolist()) != {0.0, 1.0} or not torch.isfinite(x).all():
            raise ValueError("Fit requires finite inputs and both classes")
        if validation is not None:
            vx = torch.as_tensor(validation[0], dtype=torch.float32)
            vl = torch.as_tensor(validation[1], dtype=torch.long)
            vy = torch.as_tensor(validation[2], dtype=torch.float32)
            if set(vy.tolist()) != {0.0, 1.0}:
                raise ValueError("Checkpoint people must contain both classes")
        self.network_ = TemporalNetwork(self.kind, self.config["dropout"])
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.config["learning_rate"],
                                      weight_decay=self.config["weight_decay"])
        generator = torch.Generator().manual_seed(self.seed)
        best, wait, state = float("inf"), 0, None
        self.history_ = []
        limit = self.config["max_epochs"] if validation is not None else fixed_epochs
        for epoch in range(1, limit + 1):
            self.network_.train()
            total = 0.0
            order = torch.randperm(len(y), generator=generator)
            for start in range(0, len(y), self.config["batch_size"]):
                index = order[start:start + self.config["batch_size"]]
                optimizer.zero_grad()
                loss = self._loss(x[index], lengths[index], y[index])
                loss = loss + self.config["kernel_l2"] * self.network_.kernel_penalty()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(index)
            self.network_.eval()
            row = {"epoch": epoch, "train_loss": total / len(y)}
            if validation is not None:
                with torch.inference_mode():
                    total_val = 0.0
                    for start in range(0, len(vy), self.config["batch_size"]):
                        stop = start + self.config["batch_size"]
                        total_val += float(self._loss(vx[start:stop], vl[start:stop], vy[start:stop])) * len(vy[start:stop])
                    val_loss = total_val / len(vy) + self.config["kernel_l2"] * float(self.network_.kernel_penalty())
                row["checkpoint_loss"] = val_loss
                if not np.isfinite(val_loss):
                    raise ValueError("Nonfinite checkpoint loss; experiment stops without a retry")
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

    def predict_proba(self, values, lengths):
        torch.set_num_threads(self.config["cpu_threads"])
        x = torch.as_tensor(values, dtype=torch.float32)
        lengths = torch.as_tensor(lengths, dtype=torch.long)
        if not len(x):
            raise ValueError("No complete gyro windows")
        self.network_.eval()
        with torch.inference_mode():
            scores = [torch.sigmoid(self.network_(x[start:start + self.config["batch_size"]],
                                                  lengths[start:start + self.config["batch_size"]])).numpy()
                      for start in range(0, len(x), self.config["batch_size"])]
        positive = np.concatenate(scores).astype(float)
        return np.column_stack([1 - positive, positive])


@dataclass
class TemporalFoG(A1FogModel):
    fit_participants: tuple[int, ...] = ()
    threshold_participants: tuple[int, ...] = ()

    def predict_windows(self, windows):
        if self.channel_names != GYRO_CHANNELS:
            raise ValueError("Unexpected temporal model gyro schema")
        values, lengths, _ = sequence_batch(windows)
        scores = self.classifier.predict_proba(values, lengths)[:, 1]
        frame = prediction_frame(windows, scores, self.threshold, self.variant)
        frame["model_version"] = f"a1_temporal_cnn_v1/{self.variant}"
        frame["prediction_source"] = "train14_model_train_oof_threshold"
        frame["context_windows"] = lengths if self.variant == "cnn_gru" else 1
        return frame

    def predict_frame(self, frame):
        return self.predict_windows(gyro_windows(frame.drop(columns=[
            c for c in ("fog", "activity", "fog_severity") if c in frame])))

    def new_stream(self):
        return TemporalStream(self)


class TemporalStream:
    """Consume completed 2s windows every 0.5s; buffer is per independent stream."""
    def __init__(self, model):
        self.model = model
        self.history = deque(maxlen=9)

    def push(self, window):
        if self.history:
            previous = self.history[-1]
            same_key = (window.subject_id, window.session_id, window.task_id) == (
                previous.subject_id, previous.session_id, previous.task_id)
            if same_key and window.start_s <= previous.start_s:
                raise ValueError("Stream windows must be chronological")
            if not same_key or not np.isclose(window.start_s - previous.start_s, 0.5, atol=1e-6, rtol=0):
                self.history.clear()
        self.history.append(window)
        return self.model.predict_windows(list(self.history)).iloc[-1:].reset_index(drop=True)
