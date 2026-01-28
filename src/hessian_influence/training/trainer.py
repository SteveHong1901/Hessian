from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader

from hessian_influence.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizerType(str, Enum):
    SGD = "sgd"
    ADAM = "adam"


class SchedulerType(str, Enum):
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    COSINE = "cosine"


@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.9
    max_epochs: int = 100
    optimizer_type: OptimizerType = OptimizerType.SGD
    scheduler_type: Optional[SchedulerType] = None
    scheduler_params: dict = field(default_factory=dict)
    early_stopping: bool = False
    patience: int = 10
    checkpoint_dir: Optional[Path] = None
    save_every_n_epochs: int = 10


@dataclass
class TrainingState:
    epoch: int = 0
    train_loss: float = float("inf")
    eval_loss: float = float("inf")
    best_eval_loss: float = float("inf")
    epochs_without_improvement: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self._model = model
        self._config = config
        self._device = torch.device(device)
        self._model.to(self._device)

        self._optimizer = self._create_optimizer()
        self._scheduler = self._create_scheduler()
        self._state = TrainingState()

        logger.info(f"Initialized Trainer on {self._device}")

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def state(self) -> TrainingState:
        return self._state

    def _create_optimizer(self) -> Optimizer:
        if self._config.optimizer_type == OptimizerType.SGD:
            return SGD(
                self._model.parameters(),
                lr=self._config.learning_rate,
                momentum=self._config.momentum,
                weight_decay=self._config.weight_decay,
            )
        elif self._config.optimizer_type == OptimizerType.ADAM:
            return Adam(
                self._model.parameters(),
                lr=self._config.learning_rate,
                weight_decay=self._config.weight_decay,
            )
        raise ValueError(f"Unknown optimizer type: {self._config.optimizer_type}")

    def _create_scheduler(self) -> Optional[_LRScheduler]:
        if self._config.scheduler_type is None:
            return None

        params = self._config.scheduler_params
        stype = self._config.scheduler_type

        if stype == SchedulerType.STEP:
            default_params = {"step_size": 50, "gamma": 0.2}
            default_params.update(params)
            return StepLR(self._optimizer, **default_params)

        elif stype == SchedulerType.EXPONENTIAL:
            default_params = {"gamma": 0.95}
            default_params.update(params)
            return ExponentialLR(self._optimizer, **default_params)

        elif stype == SchedulerType.PLATEAU:
            default_params = {"factor": 0.5, "patience": 10, "threshold": 1e-4, "mode": "min"}
            default_params.update(params)
            return ReduceLROnPlateau(self._optimizer, **default_params)

        elif stype == SchedulerType.COSINE:
            default_params = {"T_max": self._config.max_epochs, "eta_min": 1e-8}
            default_params.update(params)
            return CosineAnnealingLR(self._optimizer, **default_params)

        raise ValueError(f"Unknown scheduler type: {stype}")

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> nn.Module:
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(reduction="mean")

        for epoch in range(1, self._config.max_epochs + 1):
            self._state.epoch = epoch
            self._train_epoch(train_loader, loss_fn)

            if eval_loader is not None:
                self._state.eval_loss = self._evaluate(eval_loader, loss_fn)

            self._update_scheduler()
            self._log_progress()

            if self._should_save():
                self._save_checkpoint()

            if self._should_stop():
                logger.info(f"Early stopping at epoch {epoch}")
                break

        return self._model

    def _train_epoch(self, loader: DataLoader, loss_fn: nn.Module) -> None:
        self._model.train()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets in loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            self._optimizer.zero_grad()
            outputs = self._model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self._state.train_loss = total_loss / max(num_batches, 1)

    def _evaluate(self, loader: DataLoader, loss_fn: nn.Module) -> float:
        self._model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                outputs = self._model(inputs)
                loss = F.cross_entropy(outputs, targets, reduction="sum")

                total_loss += loss.item()
                total_samples += inputs.size(0)

        return total_loss / max(total_samples, 1)

    def _update_scheduler(self) -> None:
        if self._scheduler is None:
            return

        if isinstance(self._scheduler, ReduceLROnPlateau):
            self._scheduler.step(self._state.eval_loss)
        else:
            self._scheduler.step()

    def _log_progress(self) -> None:
        lr = self._optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {self._state.epoch}: "
            f"train_loss={self._state.train_loss:.4f}, "
            f"eval_loss={self._state.eval_loss:.4f}, "
            f"lr={lr:.6f}"
        )

    def _should_stop(self) -> bool:
        if not self._config.early_stopping:
            return False

        if self._state.eval_loss < self._state.best_eval_loss:
            self._state.best_eval_loss = self._state.eval_loss
            self._state.epochs_without_improvement = 0
        else:
            self._state.epochs_without_improvement += 1

        return self._state.epochs_without_improvement >= self._config.patience

    def _should_save(self) -> bool:
        if self._config.checkpoint_dir is None:
            return False
        return self._state.epoch % self._config.save_every_n_epochs == 0

    def _save_checkpoint(self) -> None:
        if self._config.checkpoint_dir is None:
            return

        self._config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._config.checkpoint_dir / f"epoch_{self._state.epoch}.pt"

        torch.save(
            {
                "epoch": self._state.epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "train_loss": self._state.train_loss,
                "eval_loss": self._state.eval_loss,
            },
            checkpoint_path,
        )
        logger.debug(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._state.epoch = checkpoint["epoch"]
        self._state.train_loss = checkpoint["train_loss"]
        self._state.eval_loss = checkpoint["eval_loss"]
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
