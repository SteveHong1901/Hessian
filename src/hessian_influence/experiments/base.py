from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hessian_influence.utils.logging import get_logger, setup_logger
from hessian_influence.utils.seed import SeedManager

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: Path = Path("outputs")
    log_level: int = 20
    save_results: bool = True
    extra: dict = field(default_factory=dict)


class BaseExperiment(ABC):
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._output_dir = config.output_dir / config.name
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._logger = setup_logger(
            config.name,
            level=config.log_level,
            log_file=self._output_dir / "experiment.log",
        )
        SeedManager.set_seed(config.seed)

        self._model: Optional[nn.Module] = None
        self._train_loader: Optional[DataLoader] = None
        self._eval_loader: Optional[DataLoader] = None
        self._valid_loader: Optional[DataLoader] = None
        self._loss_fn: Optional[nn.Module] = None
        self._results: dict[str, Any] = {}

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        return self._model

    @property
    def results(self) -> dict[str, Any]:
        return self._results

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def run(self) -> dict[str, Any]:
        pass

    def save_results(self, filename: str = "results.pt") -> Path:
        path = self._output_dir / filename
        torch.save(self._results, path)
        self._logger.info(f"Saved results to {path}")
        return path

    def load_model(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._model.eval()
        self._logger.info(f"Loaded model from {path}")

    def save_model(self, filename: str = "model.pt") -> Path:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        path = self._output_dir / filename
        torch.save(self._model.state_dict(), path)
        self._logger.info(f"Saved model to {path}")
        return path
