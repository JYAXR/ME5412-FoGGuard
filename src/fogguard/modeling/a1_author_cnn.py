"""Independent PyTorch implementation of the cited single-ankle CNN architecture.

Architecture: Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor,
model.json; Borzi et al. DOI:10.1177/1877718X241302766. No pretrained weights.
Training, labels and split scope are documented in docs/a1/SELECTED_METHODS.md.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fogguard.data.fogstar import FogStarWindow, _contiguous_runs
from fogguard.modeling.a1_fog import A1FogModel, prediction_frame

GYRO_CHANNELS = tuple(f"ankleL_gyro_{axis}" for axis in "xyz")


def gyro_windows(frame):
    """Require only real gyro samples; never cross record/gap or impute accel."""
    required = {"timestamp", *GYRO_CHANNELS}
    if not required <= set(frame):
        raise ValueError("Missing left ankle gyro/timestamp columns")
    frame = frame.copy()
    keys = ["subjectID", "sessionID", "taskID"]
    for name, default in zip(keys, ("new", "1", "0"), strict=True):
        if name not in frame:
            frame[name] = default
    blocks = frame[keys].ne(frame[keys].shift()).any(axis=1).cumsum()
    result = []
    for _, record in frame.groupby(blocks, sort=False):
        times = record.timestamp.to_numpy(float)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Record timestamps must be finite and increasing")
        valid = np.isfinite(record[list(GYRO_CHANNELS)].to_numpy(float)).all(axis=1)
        for section in _contiguous_runs(valid, times):
            run = record.iloc[section]
            for start in range(0, len(run) - 120 + 1, 30):
                chunk = run.iloc[start:start + 120]
                fraction = float(chunk.fog.mean()) if "fog" in chunk else None
                activity, activity_fraction = None, None
                if "activity" in chunk:
                    counts = chunk.activity.value_counts()
                    activity_fraction = float(counts.iloc[0] / 120)
                    if (counts == counts.iloc[0]).sum() == 1 and activity_fraction >= 0.5:
                        activity = int(counts.index[0])
                window = FogStarWindow(
                    chunk[list(GYRO_CHANNELS)].to_numpy(float), GYRO_CHANNELS, 60.0,
                    float(chunk.timestamp.iloc[0]), float(chunk.timestamp.iloc[-1]),
                    str(chunk.subjectID.iloc[0]), str(chunk.sessionID.iloc[0]),
                    str(chunk.taskID.iloc[0]),
                    fog_label=int(fraction >= 0.5) if fraction is not None else None,
                    fog_fraction=fraction, activity_label=activity,
                    activity_fraction=activity_fraction)
                window.validate()
                result.append(window)
    return result


def centered_gyro(windows):
    if not windows:
        raise ValueError("No gyro windows")
    values = []
    for window in windows:
        window.validate()
        if window.channel_names != GYRO_CHANNELS or window.sample_rate_hz != 60 or len(window.samples) != 120:
            raise ValueError("Author architecture requires 120 x 3 left ankle gyro at 60Hz")
        values.append((window.samples - window.samples.mean(axis=0)).T)
    return np.asarray(values, dtype=np.float32)


class CausalConv(nn.Module):
    def __init__(self, incoming, outgoing, kernel):
        super().__init__()
        self.left_padding = kernel - 1
        self.conv = nn.Conv1d(incoming, outgoing, kernel)

    def forward(self, values):
        return F.relu(self.conv(F.pad(values, (self.left_padding, 0))))


class AuthorFoGCNN(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.conv1, self.conv2 = CausalConv(3, 16, 9), CausalConv(16, 16, 7)
        self.conv3, self.conv4 = CausalConv(16, 24, 5), CausalConv(24, 24, 5)
        self.drop = nn.Dropout(dropout)
        self.dense, self.output = nn.Linear(24, 48), nn.Linear(48, 1)
        for layer in self.regularized_layers():
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def regularized_layers(self):
        return [self.conv1.conv, self.conv2.conv, self.conv3.conv, self.conv4.conv, self.dense]

    def kernel_penalty(self):
        return sum(layer.weight.square().sum() for layer in self.regularized_layers())

    def forward(self, values):
        values = self.conv2(self.conv1(values))
        values = self.drop(F.max_pool1d(values, 4))
        values = self.drop(F.max_pool1d(self.conv3(values), 3))
        values = self.conv4(values).mean(dim=2)
        return self.output(self.drop(F.relu(self.dense(values)))).squeeze(-1)


class AuthorCNNClassifier:
    def __init__(self, config, seed):
        self.config, self.seed = dict(config), seed

    def fit(self, values, labels, validation):
        torch.set_num_threads(self.config["cpu_threads"])
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        x, y = torch.as_tensor(values, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32)
        vx = torch.as_tensor(validation[0], dtype=torch.float32)
        vy = torch.as_tensor(validation[1], dtype=torch.float32)
        if set(y.tolist()) != {0.0, 1.0} or set(vy.tolist()) != {0.0, 1.0}:
            raise ValueError("Training and validation must contain both classes")
        self.network_ = AuthorFoGCNN(self.config["dropout"])
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.config["learning_rate"],
                                      weight_decay=self.config["weight_decay"])
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.seed)
        best, wait, state = float("inf"), 0, None
        self.history_ = []
        for epoch in range(1, self.config["max_epochs"] + 1):
            self.network_.train()
            order = torch.randperm(len(y), generator=generator)
            total = 0.0
            for start in range(0, len(y), self.config["batch_size"]):
                indices = order[start:start + self.config["batch_size"]]
                optimizer.zero_grad()
                loss = criterion(self.network_(x[indices]), y[indices])
                loss = loss + self.config["kernel_l2"] * self.network_.kernel_penalty()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            self.network_.eval()
            with torch.inference_mode():
                total_validation = 0.0
                for start in range(0, len(vy), self.config["batch_size"]):
                    stop = start + self.config["batch_size"]
                    total_validation += float(criterion(self.network_(vx[start:stop]), vy[start:stop])) * len(vy[start:stop])
                val_bce = total_validation / len(vy)
                val_loss = val_bce + self.config["kernel_l2"] * float(self.network_.kernel_penalty())
            self.history_.append({"epoch": epoch, "train_loss": total / len(y),
                                  "validation_bce": val_bce, "validation_loss": val_loss})
            if val_loss < best - self.config["min_delta"]:
                best, wait, state = val_loss, 0, copy.deepcopy(self.network_.state_dict())
                self.best_epoch_ = epoch
            else:
                wait += 1
            print(f"CNN epoch {epoch:03d}: train={total / len(y):.4f}, validation={val_loss:.4f}, checkpoint={self.best_epoch_}", flush=True)
            if wait >= self.config["patience"]:
                break
        self.network_.load_state_dict(state)
        self.network_.eval()
        self.classes_ = np.array([0, 1])
        self.parameter_count_ = sum(p.numel() for p in self.network_.parameters())
        return self

    def predict_proba(self, values):
        x = torch.as_tensor(values, dtype=torch.float32)
        if not len(x):
            raise ValueError("No gyro input")
        self.network_.eval()
        with torch.inference_mode():
            scores = [torch.sigmoid(self.network_(x[start:start + self.config["batch_size"]])).numpy()
                      for start in range(0, len(x), self.config["batch_size"])]
        positive = np.concatenate(scores).astype(float)
        return np.column_stack([1 - positive, positive])


@dataclass
class GyroFoG(A1FogModel):
    fit_participants: tuple[int, ...] = ()
    threshold_participants: tuple[int, ...] = ()

    def predict_windows(self, windows):
        if self.channel_names != GYRO_CHANNELS or any(w.channel_names != GYRO_CHANNELS for w in windows):
            raise ValueError("Unexpected deployed gyro schema")
        if self.variant == "author_cnn":
            values = centered_gyro(windows)
        else:
            raise ValueError("Unknown gyro detector variant")
        score = self.classifier.predict_proba(values)[:, 1]
        result = prediction_frame(windows, score, self.threshold, self.variant)
        result["model_version"] = f"a1_author_cnn_v1/{self.variant}"
        result["prediction_source"] = "train14_new_input"
        return result

    def predict_frame(self, frame):
        return self.predict_windows(gyro_windows(frame.drop(columns=[
            name for name in ("fog", "activity", "fog_severity") if name in frame])))
