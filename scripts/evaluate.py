#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from hessian_influence.core.hessian import HessianComputer, HessianConfig, CurvatureType
from hessian_influence.core.inversion import MatrixInverter, InversionConfig, InversionMethod
from hessian_influence.data.loaders import DataLoaderFactory
from hessian_influence.evaluation.metrics import HessianMetrics
from hessian_influence.influence.calculator import InfluenceCalculator
from hessian_influence.training.models import ModelConfig, ModelFactory, ActivationType
from hessian_influence.utils.logging import setup_logger
from hessian_influence.utils.seed import SeedManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate influence functions")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32], help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
    parser.add_argument("--curvature", type=str, default="H", help="Curvature type: H or GGN")
    parser.add_argument("--damping", type=float, default=1e-4, help="Damping for inversion")
    parser.add_argument("--eps", type=float, default=1e-6, help="Eigenvalue threshold")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = setup_logger("evaluate", log_file=args.output_dir / "evaluate.log")
    logger.info(f"Starting evaluation with args: {args}")

    SeedManager.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_config = ModelConfig(
        input_dim=64,
        output_dim=10,
        hidden_sizes=tuple(args.hidden),
        activation=ActivationType(args.activation.lower()),
    )
    model = ModelFactory.create_mlp(model_config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Loaded model")

    loader_factory = DataLoaderFactory()
    _, eval_train_loader, valid_loader = loader_factory.create_loaders()

    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    hessian_config = HessianConfig(
        curvature_type=CurvatureType(args.curvature),
        damping=args.damping,
    )
    hessian_computer = HessianComputer(model, loss_fn, eval_train_loader, hessian_config)

    logger.info(f"Computing {args.curvature} operator...")
    operator = hessian_computer.compute_operator()
    logger.info(f"Operator shape: {operator.shape}")

    metrics = HessianMetrics()
    stats = metrics.stability_statistics(operator, verbose=True)
    logger.info(f"Condition number: {stats.condition_number:.2e}")

    inversion_config = InversionConfig(
        method=InversionMethod.EIGEN,
        damping=args.damping,
        eps=args.eps,
    )
    inverter = MatrixInverter(inversion_config)

    logger.info("Computing inverse Hessian...")
    inverse = inverter.invert(operator)
    inverse_tensor = torch.from_numpy(inverse).to(device).float()

    influence_calc = InfluenceCalculator(device=device)

    logger.info("Computing gradients...")
    train_grads = influence_calc.compute_sample_gradients(model, eval_train_loader, loss_fn)
    valid_grads = influence_calc.compute_sample_gradients(model, valid_loader, loss_fn)

    logger.info("Computing influence scores...")
    scores = influence_calc.compute_influence_scores_batch(valid_grads, train_grads, inverse_tensor)

    output_path = args.output_dir / "influence_scores.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scores": scores, "stats": stats}, output_path)
    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
