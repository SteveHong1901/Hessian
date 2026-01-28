from dataclasses import dataclass
from typing import Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from hessian_influence.data.datasets import DatasetConfig, DatasetFactory


@dataclass
class LoaderConfig:
    train_batch_size: int = 32
    eval_batch_size: int = 4096
    num_workers: int = 0
    pin_memory: bool = False


class DataLoaderFactory:
    def __init__(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        loader_config: Optional[LoaderConfig] = None,
    ) -> None:
        self._dataset_config = dataset_config or DatasetConfig()
        self._loader_config = loader_config or LoaderConfig()
        self._dataset_factory = DatasetFactory(self._dataset_config)

    def create_loaders(
        self,
        train_indices: Optional[list[int]] = None,
        valid_indices: Optional[list[int]] = None,
        apply_corruption: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset = self._dataset_factory.create_digits_dataset(
            split="train",
            indices=train_indices,
            apply_corruption=apply_corruption,
        )
        eval_train_dataset = self._dataset_factory.create_digits_dataset(
            split="eval_train",
            indices=train_indices,
            apply_corruption=apply_corruption,
        )
        valid_dataset = self._dataset_factory.create_digits_dataset(
            split="valid",
            indices=valid_indices,
            apply_corruption=False,
        )

        train_loader = self._create_loader(
            train_dataset,
            batch_size=self._loader_config.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        eval_train_loader = self._create_loader(
            eval_train_dataset,
            batch_size=self._loader_config.eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
        valid_loader = self._create_loader(
            valid_dataset,
            batch_size=self._loader_config.eval_batch_size,
            shuffle=False,
            drop_last=False,
        )

        return train_loader, eval_train_loader, valid_loader

    def _create_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self._loader_config.num_workers,
            pin_memory=self._loader_config.pin_memory,
        )
