from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class DatasetName(str, Enum):
    DIGITS = "digits"
    UCI_CONCRETE = "concrete"
    UCI_PARKINSONS = "parkinsons"
    SYNTHETIC = "synthetic"
    XOR = "xor"


@dataclass
class PipelineConfig:
    dataset_name: DatasetName = DatasetName.DIGITS
    train_ratio: float = 0.9
    train_batch_size: int = 32
    eval_batch_size: int = 4096
    corruption_ratio: float = 0.0
    random_seed: int = 0
    data_path: Optional[Path] = None


class TensorDataset(Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray) -> None:
        self._x = data_x
        self._y = data_y

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._x[index], self._y[index]

    def __len__(self) -> int:
        return len(self._x)


class ExperimentPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._rng = np.random.RandomState(config.random_seed)

    def get_model(self, hidden_sizes: Optional[list[int]] = None, activation: str = "gelu") -> nn.Module:
        dataset_info = self._get_dataset_info()
        input_dim = dataset_info["input_dim"]
        output_dim = dataset_info["output_dim"]

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        act_cls = activation_map.get(activation.lower(), nn.GELU)

        layers: list[nn.Module] = []
        in_features = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(in_features, hs, bias=True))
            layers.append(act_cls())
            in_features = hs
        layers.append(nn.Linear(in_features, output_dim, bias=True))

        return nn.Sequential(*layers)

    def get_loaders(
        self,
        train_indices: Optional[list[int]] = None,
        valid_indices: Optional[list[int]] = None,
        apply_corruption: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        x_train, y_train, x_valid, y_valid = self._load_data()

        if apply_corruption and self._config.corruption_ratio > 0:
            y_train = self._apply_corruption(y_train)

        train_dataset = TensorDataset(x_train, y_train)
        eval_train_dataset = TensorDataset(x_train, y_train)
        valid_dataset = TensorDataset(x_valid, y_valid)

        if train_indices is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            eval_train_dataset = torch.utils.data.Subset(eval_train_dataset, train_indices)
        if valid_indices is not None:
            valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self._config.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        eval_train_loader = DataLoader(
            eval_train_dataset,
            batch_size=self._config.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self._config.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        return train_loader, eval_train_loader, valid_loader

    def get_loss_fn(self) -> nn.Module:
        dataset_info = self._get_dataset_info()
        if dataset_info["task"] == "classification":
            return nn.CrossEntropyLoss(reduction="mean")
        return nn.MSELoss(reduction="mean")

    def get_hyperparameters(self) -> dict:
        hyperparams = {
            DatasetName.DIGITS: {"lr": 0.03, "wd": 0.0},
            DatasetName.UCI_CONCRETE: {"lr": 0.03, "wd": 0.0},
            DatasetName.UCI_PARKINSONS: {"lr": 0.01, "wd": 0.0},
            DatasetName.SYNTHETIC: {"lr": 0.01, "wd": 0.0},
            DatasetName.XOR: {"lr": 0.1, "wd": 0.0},
        }
        return hyperparams.get(self._config.dataset_name, {"lr": 0.01, "wd": 0.0})

    def _get_dataset_info(self) -> dict:
        info_map = {
            DatasetName.DIGITS: {"input_dim": 64, "output_dim": 10, "task": "classification"},
            DatasetName.UCI_CONCRETE: {"input_dim": 8, "output_dim": 1, "task": "regression"},
            DatasetName.UCI_PARKINSONS: {"input_dim": 21, "output_dim": 1, "task": "regression"},
            DatasetName.SYNTHETIC: {"input_dim": 10, "output_dim": 2, "task": "classification"},
            DatasetName.XOR: {"input_dim": 2, "output_dim": 2, "task": "classification"},
        }
        return info_map[self._config.dataset_name]

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._config.dataset_name == DatasetName.DIGITS:
            return self._load_digits()
        elif self._config.dataset_name in [DatasetName.UCI_CONCRETE, DatasetName.UCI_PARKINSONS]:
            return self._load_uci()
        elif self._config.dataset_name == DatasetName.SYNTHETIC:
            return self._load_synthetic()
        elif self._config.dataset_name == DatasetName.XOR:
            return self._load_xor()
        raise ValueError(f"Unknown dataset: {self._config.dataset_name}")

    def _load_digits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        digits = load_digits()
        X = digits.data.astype(np.float32)
        y = digits.target.astype(np.int64)

        permutation = self._rng.choice(len(X), len(X), replace=False)
        split_idx = int(len(X) * self._config.train_ratio)

        x_train = X[permutation[:split_idx]]
        y_train = y[permutation[:split_idx]]
        x_valid = X[permutation[split_idx:]]
        y_valid = y[permutation[split_idx:]]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_valid = scaler.transform(x_valid).astype(np.float32)

        return x_train, y_train, x_valid, y_valid

    def _load_uci(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._config.data_path is None:
            raise ValueError("data_path must be set for UCI datasets")

        filename = f"{self._config.dataset_name.value}.data"
        data = np.loadtxt(self._config.data_path / filename, delimiter=None)
        data = data.astype(np.float32)

        permutation = self._rng.choice(len(data), len(data), replace=False)
        split_idx = int(len(data) * self._config.train_ratio)

        train_data = data[permutation[:split_idx]]
        valid_data = data[permutation[split_idx:]]

        x_train, y_train = train_data[:, :-1], train_data[:, -1:]
        x_valid, y_valid = valid_data[:, :-1], valid_data[:, -1:]

        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train).astype(np.float32)
        x_valid = x_scaler.transform(x_valid).astype(np.float32)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train).astype(np.float32)
        y_valid = y_scaler.transform(y_valid).astype(np.float32)

        return x_train, y_train, x_valid, y_valid

    def _load_synthetic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = 1000
        n_features = 10
        n_classes = 2

        X = self._rng.randn(n_samples, n_features).astype(np.float32)
        weights = self._rng.randn(n_features, n_classes).astype(np.float32)
        logits = X @ weights + 0.1 * self._rng.randn(n_samples, n_classes).astype(np.float32)
        y = np.argmax(logits, axis=1).astype(np.int64)

        split_idx = int(n_samples * self._config.train_ratio)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    def _load_xor(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = 500
        X = self._rng.randn(n_samples, 2).astype(np.float32)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)

        split_idx = int(n_samples * self._config.train_ratio)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    def _apply_corruption(self, labels: np.ndarray) -> np.ndarray:
        num_corrupt = int(len(labels) * self._config.corruption_ratio)
        if num_corrupt == 0:
            return labels

        labels = labels.copy()
        dataset_info = self._get_dataset_info()

        if dataset_info["task"] == "classification":
            num_classes = dataset_info["output_dim"]
            labels[:num_corrupt] = self._rng.randint(0, num_classes, num_corrupt)
        else:
            labels[:num_corrupt] = self._rng.randn(num_corrupt, 1).astype(np.float32)

        return labels
