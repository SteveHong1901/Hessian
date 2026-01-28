#!/usr/bin/env python
import argparse
from pathlib import Path

import torch

from hessian_influence.data.loaders import DataLoaderFactory, LoaderConfig
from hessian_influence.data.datasets import DatasetConfig
from hessian_influence.training.models import ModelConfig, ModelFactory, ActivationType
from hessian_influence.training.trainer import Trainer, TrainingConfig, SchedulerType
from hessian_influence.utils.logging import setup_logger
from hessian_influence.utils.seed import SeedManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32], help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
    parser.add_argument("--scheduler", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = setup_logger("train", log_file=args.output_dir / "train.log")
    logger.info(f"Starting training with args: {args}")

    SeedManager.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_config = DatasetConfig(train_ratio=0.9)
    loader_config = LoaderConfig(train_batch_size=args.batch_size)
    loader_factory = DataLoaderFactory(dataset_config, loader_config)

    train_loader, eval_train_loader, valid_loader = loader_factory.create_loaders()

    model_config = ModelConfig(
        input_dim=64,
        output_dim=10,
        hidden_sizes=tuple(args.hidden),
        activation=ActivationType(args.activation.lower()),
    )
    model = ModelFactory.create_mlp(model_config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    training_config = TrainingConfig(
        learning_rate=args.lr,
        max_epochs=args.epochs,
        scheduler_type=SchedulerType(args.scheduler.lower()) if args.scheduler else None,
        checkpoint_dir=args.output_dir / "checkpoints",
    )
    trainer = Trainer(model, training_config, device=device)

    model = trainer.train(train_loader, valid_loader)

    model_path = args.output_dir / "model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
