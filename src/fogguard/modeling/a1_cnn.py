"""Small CPU CNN trained from scratch; import only for the optional CNN arm."""

import numpy as np
import torch
from torch import nn


class TinyFoGCNN(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        layers = []
        previous = input_channels
        for index, (channels, kernel) in enumerate(zip(config["channels"], config["kernels"],
                                                     strict=True)):
            layers.extend([nn.Conv1d(previous, channels, kernel, padding=kernel // 2), nn.ReLU()])
            if index < 2:
                layers.append(nn.MaxPool1d(2))
            previous = channels
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                     nn.Dropout(config["dropout"]), nn.Linear(previous, 1))

    def forward(self, values):
        return self.network(values).squeeze(-1)


class TorchWindowClassifier:
    def __init__(self, config, seed):
        self.config, self.seed = dict(config), seed

    def fit(self, values, labels):
        torch.set_num_threads(self.config["cpu_threads"])
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        values = np.asarray(values, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if set(labels) != {0.0, 1.0}:
            raise ValueError("CNN training requires both classes")
        self.mean_ = values.mean(axis=(0, 2), keepdims=True)
        self.scale_ = np.maximum(values.std(axis=(0, 2), keepdims=True), 1e-6)
        x = torch.from_numpy((values - self.mean_) / self.scale_)
        y = torch.from_numpy(labels)
        self.network_ = TinyFoGCNN(values.shape[1], self.config)
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.config["learning_rate"],
                                     weight_decay=self.config["weight_decay"])
        weight = float((labels == 0).sum() / (labels == 1).sum())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history_ = []
        self.network_.train()
        for _ in range(self.config["epochs"]):
            order = torch.randperm(len(y), generator=generator)
            total = 0.0
            for start in range(0, len(y), self.config["batch_size"]):
                indices = order[start:start + self.config["batch_size"]]
                optimizer.zero_grad()
                loss = criterion(self.network_(x[indices]), y[indices])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            self.loss_history_.append(total / len(y))
        self.network_.eval()
        self.classes_ = np.array([0, 1])
        self.parameter_count_ = sum(p.numel() for p in self.network_.parameters())
        return self

    def predict_proba(self, values):
        values = np.asarray(values, dtype=np.float32)
        x = torch.from_numpy((values - self.mean_) / self.scale_)
        self.network_.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(x), self.config["batch_size"]):
                scores.append(torch.sigmoid(self.network_(x[start:start + self.config[
                    "batch_size"]])).numpy())
        positive = np.concatenate(scores).astype(float)
        return np.column_stack([1 - positive, positive])
