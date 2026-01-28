from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class DatasetType(str, Enum):
    DIGITS = "digits"
    SYNTHETIC = "synthetic"


@dataclass
class DatasetConfig:
    dataset_type: DatasetType = DatasetType.DIGITS
    train_ratio: float = 0.9
    corruption_ratio: float = 0.0
    random_seed: int = 0


class BaseDataset(Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray) -> None:
        self._data_x = data_x
        self._data_y = data_y

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._data_x[index], self._data_y[index]

    def __len__(self) -> int:
        return self._data_x.shape[0]

    @property
    def input_dim(self) -> int:
        return self._data_x.shape[1]

    @property
    def num_classes(self) -> int:
        return len(np.unique(self._data_y))


class DatasetFactory:
    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self._config = config or DatasetConfig()
        self._rng = np.random.RandomState(self._config.random_seed)

    def create_digits_dataset(
        self,
        split: str,
        indices: Optional[list[int]] = None,
        apply_corruption: bool = False,
    ) -> BaseDataset:
        digits = load_digits()
        X = digits.data.astype(np.float32)
        y = digits.target.astype(np.int64)

        permutation = self._rng.choice(np.arange(len(X)), len(X), replace=False)
        split_idx = int(np.round(len(X) * self._config.train_ratio))

        train_indices = permutation[:split_idx]
        valid_indices = permutation[split_idx:]

        x_train, y_train = X[train_indices], y[train_indices]
        x_valid, y_valid = X[valid_indices], y[valid_indices]

        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
        x_valid_scaled = scaler.transform(x_valid).astype(np.float32)

        if apply_corruption and split in ["train", "eval_train"]:
            y_train = self._apply_corruption(y_train)

        if split in ["train", "eval_train"]:
            data_x, data_y = x_train_scaled, y_train
        else:
            data_x, data_y = x_valid_scaled, y_valid

        dataset = BaseDataset(data_x, data_y)

        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)

        return dataset

    def create_synthetic_dataset(
        self,
        num_samples: int,
        input_dim: int,
        num_classes: int,
        noise_std: float = 0.1,
    ) -> BaseDataset:
        X = self._rng.randn(num_samples, input_dim).astype(np.float32)
        true_weights = self._rng.randn(input_dim, num_classes).astype(np.float32)
        logits = X @ true_weights + noise_std * self._rng.randn(
            num_samples, num_classes
        ).astype(np.float32)
        y = np.argmax(logits, axis=1).astype(np.int64)

        return BaseDataset(X, y)

    def _apply_corruption(self, labels: np.ndarray) -> np.ndarray:
        if self._config.corruption_ratio <= 0:
            return labels

        num_corrupt = int(np.ceil(len(labels) * self._config.corruption_ratio))
        num_classes = len(np.unique(labels))

        generator = torch.Generator().manual_seed(self._config.random_seed)
        new_labels = torch.randint(0, num_classes, (num_corrupt,), generator=generator)
        labels[:num_corrupt] = new_labels.numpy()

        return labels
