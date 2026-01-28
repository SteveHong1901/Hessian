from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from hessian_influence.core.hessian import CurvatureType, HessianComputer, HessianConfig
from hessian_influence.core.inversion import InversionConfig, InversionMethod, MatrixInverter
from hessian_influence.data.loaders import DataLoaderFactory, LoaderConfig
from hessian_influence.data.datasets import DatasetConfig
from hessian_influence.evaluation.metrics import HessianMetrics
from hessian_influence.experiments.base import BaseExperiment, ExperimentConfig
from hessian_influence.influence.calculator import InfluenceCalculator
from hessian_influence.influence.lds import LDSEvaluator
from hessian_influence.training.models import ModelConfig, ModelFactory
from hessian_influence.training.trainer import Trainer, TrainingConfig


@dataclass
class HessianAnalysisConfig(ExperimentConfig):
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    curvature_types: list[CurvatureType] = field(
        default_factory=lambda: [CurvatureType.HESSIAN, CurvatureType.GGN]
    )
    damping_values: list[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    eps_values: list[float] = field(default_factory=lambda: [1e-6, 1e-4])
    compute_block_diagonal: bool = True


class HessianAnalysisExperiment(BaseExperiment):
    def __init__(self, config: HessianAnalysisConfig) -> None:
        super().__init__(config)
        self._exp_config = config

    def setup(self) -> None:
        loader_factory = DataLoaderFactory()
        self._train_loader, self._eval_loader, self._valid_loader = (
            loader_factory.create_loaders()
        )

        self._model = ModelFactory.create_mlp(self._exp_config.model_config)
        self._model.to(self._device)
        self._loss_fn = nn.CrossEntropyLoss(reduction="mean")

        self._logger.info(f"Model parameters: {sum(p.numel() for p in self._model.parameters())}")

    def run(self) -> dict[str, Any]:
        trainer = Trainer(self._model, self._exp_config.training_config, device=self._device)
        self._model = trainer.train(self._train_loader, self._valid_loader, self._loss_fn)

        params = [p for p in self._model.parameters() if p.requires_grad]
        block_sizes = [p.numel() for p in params]

        metrics = HessianMetrics()
        inverter = MatrixInverter()

        for ctype in self._exp_config.curvature_types:
            hessian_config = HessianConfig(curvature_type=ctype)
            computer = HessianComputer(
                self._model, self._loss_fn, self._eval_loader, hessian_config
            )

            operator = computer.compute_operator()
            stats = metrics.stability_statistics(operator, verbose=False)

            self._results[f"{ctype.value}_stats"] = {
                "condition_number": stats.condition_number,
                "min_eigenvalue": stats.min_eigenvalue,
                "max_eigenvalue": stats.max_eigenvalue,
                "num_positive": stats.num_positive,
                "num_negative": stats.num_negative,
            }

            for damping in self._exp_config.damping_values:
                for eps in self._exp_config.eps_values:
                    inv = inverter.invert(operator, damping=damping, eps=eps)
                    mat = inverter.to_numpy(operator)
                    identity = np.eye(mat.shape[0])
                    error = float(np.linalg.norm(mat @ inv - identity))

                    key = f"{ctype.value}_d{damping}_e{eps}"
                    self._results[key] = {"approx_error": error}

            if self._exp_config.compute_block_diagonal:
                block_op = computer.compute_block_diagonal(block_sizes, ctype)
                off_block = metrics.off_block_energy_ratio(operator, block_sizes)
                self._results[f"{ctype.value}_off_block_ratio"] = off_block

        if self._exp_config.save_results:
            self.save_results()
            self.save_model()

        return self._results


@dataclass
class InfluenceConfig(ExperimentConfig):
    model_config: ModelConfig = field(default_factory=ModelConfig)
    model_path: Optional[Path] = None
    curvature_type: CurvatureType = CurvatureType.HESSIAN
    damping: float = 1e-4
    eps: float = 1e-6


class InfluenceExperiment(BaseExperiment):
    def __init__(self, config: InfluenceConfig) -> None:
        super().__init__(config)
        self._exp_config = config

    def setup(self) -> None:
        loader_factory = DataLoaderFactory()
        self._train_loader, self._eval_loader, self._valid_loader = (
            loader_factory.create_loaders()
        )

        self._model = ModelFactory.create_mlp(self._exp_config.model_config)
        self._model.to(self._device)
        self._loss_fn = nn.CrossEntropyLoss(reduction="mean")

        if self._exp_config.model_path is not None:
            self.load_model(self._exp_config.model_path)

    def run(self) -> dict[str, Any]:
        hessian_config = HessianConfig(
            curvature_type=self._exp_config.curvature_type,
            damping=self._exp_config.damping,
        )
        computer = HessianComputer(
            self._model, self._loss_fn, self._eval_loader, hessian_config
        )

        self._logger.info("Computing Hessian operator...")
        operator = computer.compute_operator()

        inversion_config = InversionConfig(
            method=InversionMethod.EIGEN,
            damping=self._exp_config.damping,
            eps=self._exp_config.eps,
        )
        inverter = MatrixInverter(inversion_config)

        self._logger.info("Inverting Hessian...")
        inverse = inverter.invert(operator)
        inverse_tensor = torch.from_numpy(inverse).to(self._device).float()

        calculator = InfluenceCalculator(device=self._device)

        self._logger.info("Computing gradients...")
        train_grads = calculator.compute_sample_gradients(
            self._model, self._eval_loader, self._loss_fn
        )
        valid_grads = calculator.compute_sample_gradients(
            self._model, self._valid_loader, self._loss_fn
        )

        self._logger.info("Computing influence scores...")
        scores = calculator.compute_influence_scores_batch(
            valid_grads, train_grads, inverse_tensor
        )

        self._results["influence_scores"] = scores.cpu()
        self._results["train_grads_norm"] = train_grads.norm(dim=1).cpu()
        self._results["valid_grads_norm"] = valid_grads.norm(dim=1).cpu()

        if self._exp_config.save_results:
            self.save_results()

        return self._results


@dataclass
class LDSConfig(ExperimentConfig):
    model_config: ModelConfig = field(default_factory=ModelConfig)
    model_path: Optional[Path] = None
    curvature_type: CurvatureType = CurvatureType.HESSIAN
    damping: float = 1e-4
    eps: float = 1e-6
    lso_path: Path = Path("data/lso_scores")
    alpha_values: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    num_masks: int = 100


class LDSExperiment(BaseExperiment):
    def __init__(self, config: LDSConfig) -> None:
        super().__init__(config)
        self._exp_config = config

    def setup(self) -> None:
        loader_factory = DataLoaderFactory()
        self._train_loader, self._eval_loader, self._valid_loader = (
            loader_factory.create_loaders()
        )

        self._model = ModelFactory.create_mlp(self._exp_config.model_config)
        self._model.to(self._device)
        self._loss_fn = nn.CrossEntropyLoss(reduction="mean")

        if self._exp_config.model_path is not None:
            self.load_model(self._exp_config.model_path)

    def run(self) -> dict[str, Any]:
        hessian_config = HessianConfig(
            curvature_type=self._exp_config.curvature_type,
            damping=self._exp_config.damping,
        )
        computer = HessianComputer(
            self._model, self._loss_fn, self._eval_loader, hessian_config
        )

        operator = computer.compute_operator()

        inverter = MatrixInverter()
        inverse = inverter.invert(
            operator,
            damping=self._exp_config.damping,
            eps=self._exp_config.eps,
        )
        inverse_tensor = torch.from_numpy(inverse).to(self._device).float()

        calculator = InfluenceCalculator(device=self._device)
        train_grads = calculator.compute_sample_gradients(
            self._model, self._eval_loader, self._loss_fn
        )
        valid_grads = calculator.compute_sample_gradients(
            self._model, self._valid_loader, self._loss_fn
        )

        scores = calculator.compute_influence_scores_batch(
            valid_grads, train_grads, inverse_tensor
        )

        lds_evaluator = LDSEvaluator(
            lso_path=self._exp_config.lso_path,
            num_masks=self._exp_config.num_masks,
        )

        lds_results = lds_evaluator.evaluate(
            data_name="digits",
            influence_scores=scores,
            alpha_values=self._exp_config.alpha_values,
        )

        for alpha, result in lds_results.items():
            self._results[f"lds_alpha_{alpha}"] = {
                "mean": result.mean,
                "ci": result.confidence_interval,
            }
            self._logger.info(f"LDS (alpha={alpha}): {result.mean:.4f} +/- {result.confidence_interval:.4f}")

        if self._exp_config.save_results:
            self.save_results()

        return self._results
