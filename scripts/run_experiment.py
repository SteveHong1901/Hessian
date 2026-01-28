#!/usr/bin/env python
import argparse
from pathlib import Path

import torch

from hessian_influence.core.hessian import CurvatureType
from hessian_influence.experiments.runners import (
    HessianAnalysisConfig,
    HessianAnalysisExperiment,
    InfluenceConfig,
    InfluenceExperiment,
    LDSConfig,
    LDSExperiment,
)
from hessian_influence.training.models import ActivationType, ModelConfig
from hessian_influence.training.trainer import SchedulerType, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["hessian", "influence", "lds"],
        help="Experiment type to run",
    )
    parser.add_argument("--name", type=str, default="experiment", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")

    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32], help="Hidden sizes")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--scheduler", type=str, default="cosine", help="LR scheduler")

    parser.add_argument("--curvature", type=str, default="H", help="Curvature type: H or GGN")
    parser.add_argument("--damping", type=float, default=1e-4, help="Damping for inversion")
    parser.add_argument("--eps", type=float, default=1e-6, help="Eigenvalue threshold")

    parser.add_argument("--model-path", type=Path, default=None, help="Path to trained model")
    parser.add_argument("--lso-path", type=Path, default=None, help="Path to LSO scores")

    return parser.parse_args()


def run_hessian_analysis(args: argparse.Namespace) -> None:
    model_config = ModelConfig(
        hidden_sizes=tuple(args.hidden),
        activation=ActivationType(args.activation.lower()),
    )
    training_config = TrainingConfig(
        learning_rate=args.lr,
        max_epochs=args.epochs,
        scheduler_type=SchedulerType(args.scheduler.lower()) if args.scheduler else None,
    )
    config = HessianAnalysisConfig(
        name=args.name,
        seed=args.seed,
        output_dir=args.output_dir,
        model_config=model_config,
        training_config=training_config,
        curvature_types=[CurvatureType.HESSIAN, CurvatureType.GGN],
        damping_values=[1e-4, 1e-3, 1e-2],
        eps_values=[1e-6, 1e-4],
    )

    experiment = HessianAnalysisExperiment(config)
    experiment.setup()
    results = experiment.run()

    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")


def run_influence(args: argparse.Namespace) -> None:
    model_config = ModelConfig(
        hidden_sizes=tuple(args.hidden),
        activation=ActivationType(args.activation.lower()),
    )
    config = InfluenceConfig(
        name=args.name,
        seed=args.seed,
        output_dir=args.output_dir,
        model_config=model_config,
        model_path=args.model_path,
        curvature_type=CurvatureType(args.curvature),
        damping=args.damping,
        eps=args.eps,
    )

    experiment = InfluenceExperiment(config)
    experiment.setup()
    results = experiment.run()

    print(f"\nInfluence scores shape: {results['influence_scores'].shape}")


def run_lds(args: argparse.Namespace) -> None:
    model_config = ModelConfig(
        hidden_sizes=tuple(args.hidden),
        activation=ActivationType(args.activation.lower()),
    )
    config = LDSConfig(
        name=args.name,
        seed=args.seed,
        output_dir=args.output_dir,
        model_config=model_config,
        model_path=args.model_path,
        curvature_type=CurvatureType(args.curvature),
        damping=args.damping,
        eps=args.eps,
        lso_path=args.lso_path or Path("data/lso_scores"),
        alpha_values=[0.3, 0.5, 0.7, 0.9],
    )

    experiment = LDSExperiment(config)
    experiment.setup()
    results = experiment.run()

    print("\nLDS Results:")
    for key, value in results.items():
        if key.startswith("lds_"):
            print(f"  {key}: mean={value['mean']:.4f}, ci={value['ci']:.4f}")


def main() -> None:
    args = parse_args()

    if args.experiment == "hessian":
        run_hessian_analysis(args)
    elif args.experiment == "influence":
        run_influence(args)
    elif args.experiment == "lds":
        run_lds(args)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")


if __name__ == "__main__":
    main()
