# Hessian Influence

Official code for the paper **"Better Hessians Matter: Studying the Impact of Curvature Approximations in Influence Functions"**.

**Spotlight at NeurIPS 2025 Workshop on Mechanistic Interpretability**

[[arXiv]](https://arxiv.org/abs/2509.23437) [[PDF]](https://arxiv.org/pdf/2509.23437)

## Abstract

Influence functions offer a principled way to trace model predictions back to training data, but their use in deep learning is hampered by the need to invert a large, ill-conditioned Hessian matrix. Approximations such as Generalised Gauss-Newton (GGN) and Kronecker-Factored Approximate Curvature (K-FAC) have been proposed to make influence computation tractable, yet it remains unclear how the departure from exactness impacts data attribution performance. In this paper, we investigate the effect of Hessian approximation quality on influence-function attributions in a controlled classification setting. Our experiments show that better Hessian approximations consistently yield better influence score quality. We further decompose the approximation steps for recent Hessian approximation methods and evaluate each step's influence on attribution accuracy.

## Overview

This library provides efficient tools for:

- Computing exact and approximate Hessian/GGN matrices
- Various matrix inversion strategies (exact, pseudo-inverse, eigenvalue-based)
- Influence function computation for training data attribution
- Linear Datamodeling Score (LDS) evaluation
- KFAC and EKFAC approximations via curvlinops

## Installation

### Using uv (recommended)

```bash
uv sync
```

### Using pip

```bash
pip install -e .
```

### Development installation

```bash
uv sync --all-extras
```

## Project Structure

```
hessian-influence/
├── src/hessian_influence/
│   ├── core/              # Hessian computation and inversion
│   │   ├── hessian.py     # HessianComputer class
│   │   └── inversion.py   # MatrixInverter class
│   ├── evaluation/        # Metrics and evaluation
│   │   └── metrics.py     # HessianMetrics class
│   ├── experiments/       # Experiment framework
│   │   ├── base.py        # BaseExperiment class
│   │   ├── pipelines.py   # ExperimentPipeline class
│   │   └── runners.py     # Experiment runner classes
│   ├── influence/         # Influence function computation
│   │   ├── calculator.py  # InfluenceCalculator class
│   │   └── lds.py         # LDSEvaluator class
│   ├── data/              # Dataset handling
│   │   ├── datasets.py    # DatasetFactory class
│   │   └── loaders.py     # DataLoaderFactory class
│   ├── training/          # Model training
│   │   ├── models.py      # ModelFactory class
│   │   └── trainer.py     # Trainer class
│   ├── visualization/     # Plotting utilities
│   │   └── plots.py       # HessianVisualizer class
│   └── utils/             # Utility functions
│       ├── logging.py     # Logging setup
│       └── seed.py        # Reproducibility utilities
├── scripts/               # Runnable scripts
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Evaluation script
│   └── run_experiment.py  # Experiment runner
├── tests/                 # Unit tests
└── pyproject.toml         # Project configuration
```

## Quick Start

### Training a Model

```python
from hessian_influence.data.loaders import DataLoaderFactory
from hessian_influence.training.models import ModelConfig, ModelFactory
from hessian_influence.training.trainer import Trainer, TrainingConfig

loader_factory = DataLoaderFactory()
train_loader, eval_loader, valid_loader = loader_factory.create_loaders()

model_config = ModelConfig(hidden_sizes=(64, 32))
model = ModelFactory.create_mlp(model_config)

training_config = TrainingConfig(learning_rate=0.01, max_epochs=100)
trainer = Trainer(model, training_config, device="cuda")
model = trainer.train(train_loader, valid_loader)
```

### Computing Influence Scores

```python
from hessian_influence.core.hessian import HessianComputer, HessianConfig
from hessian_influence.core.inversion import MatrixInverter
from hessian_influence.influence.calculator import InfluenceCalculator

hessian_computer = HessianComputer(model, loss_fn, data_loader)
hessian_operator = hessian_computer.compute_operator()

inverter = MatrixInverter()
inverse_hessian = inverter.invert(hessian_operator, damping=1e-4)

calculator = InfluenceCalculator(device="cuda")
train_grads = calculator.compute_sample_gradients(model, train_loader, loss_fn)
test_grads = calculator.compute_sample_gradients(model, test_loader, loss_fn)

scores = calculator.compute_influence_scores_batch(
    test_grads, train_grads, torch.from_numpy(inverse_hessian)
)
```

### Evaluating Hessian Approximations

```python
from hessian_influence.evaluation.metrics import HessianMetrics

metrics = HessianMetrics()

relative_error = metrics.relative_residual(hessian, ggn)
condition = metrics.condition_number(hessian)
stats = metrics.stability_statistics(hessian, verbose=True)
```

## Command Line Interface

### Train a model

```bash
uv run python scripts/train.py --lr 0.01 --epochs 100 --hidden 64 32 --activation gelu
```

### Evaluate influence functions

```bash
uv run python scripts/evaluate.py --model-path outputs/model.pt --curvature H --damping 1e-4
```

### Run experiments

```bash
# Hessian analysis experiment
uv run python scripts/run_experiment.py --experiment hessian --name hessian_analysis

# Influence function experiment
uv run python scripts/run_experiment.py --experiment influence --model-path outputs/model.pt

# LDS evaluation experiment
uv run python scripts/run_experiment.py --experiment lds --model-path outputs/model.pt --lso-path data/lso_scores
```

## Key Features

### Curvature Types

- **Hessian (H)**: Exact second-order derivatives
- **GGN**: Generalized Gauss-Newton approximation
- **KFAC**: Kronecker-factored approximate curvature
- **EKFAC**: Eigenvalue-corrected KFAC

### Inversion Methods

- **Exact**: Direct matrix inversion
- **Pseudo**: SVD-based pseudo-inverse with truncation
- **Eigen**: Eigenvalue decomposition with configurable thresholding

### Evaluation Metrics

- Frobenius norm and relative residual
- Off-block energy ratio
- Eigenvalue and eigenbasis overlap
- Condition number and stability statistics

## Running Tests

```bash
uv run pytest tests/ -v
```

## Configuration

All components use dataclass-based configuration for type safety and clarity:

```python
from hessian_influence.core.hessian import HessianConfig, CurvatureType

config = HessianConfig(
    curvature_type=CurvatureType.GGN,
    num_probes=1000,
    damping=1e-4,
)
```

## Experiments Framework

Create reproducible experiments using the built-in framework:

```python
from hessian_influence.experiments.runners import (
    HessianAnalysisConfig,
    HessianAnalysisExperiment,
)
from hessian_influence.training.models import ModelConfig

config = HessianAnalysisConfig(
    name="my_experiment",
    seed=42,
    model_config=ModelConfig(hidden_sizes=(64, 32)),
    damping_values=[1e-4, 1e-3],
)

experiment = HessianAnalysisExperiment(config)
experiment.setup()
results = experiment.run()
```

### Dataset Pipelines

Use the pipeline system for different datasets:

```python
from hessian_influence.experiments.pipelines import (
    DatasetName,
    ExperimentPipeline,
    PipelineConfig,
)

config = PipelineConfig(
    dataset_name=DatasetName.DIGITS,
    train_ratio=0.9,
    train_batch_size=32,
)

pipeline = ExperimentPipeline(config)
train_loader, eval_loader, valid_loader = pipeline.get_loaders()
model = pipeline.get_model(hidden_sizes=[64, 32], activation="gelu")
loss_fn = pipeline.get_loss_fn()
```

### Supported Datasets

- **DIGITS**: Scikit-learn digits dataset (classification)
- **UCI_CONCRETE**: UCI concrete dataset (regression)
- **UCI_PARKINSONS**: UCI Parkinsons dataset (regression)
- **SYNTHETIC**: Synthetic classification data
- **XOR**: XOR classification task

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- curvlinops-for-pytorch >= 2.0

## License

MIT License

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{hong2025better,
  title={Better Hessians Matter: Studying the Impact of Curvature Approximations in Influence Functions},
  author={Hong, Steve and Eschenhagen, Runa and Mlodozeniec, Bruno and Turner, Richard},
  journal={arXiv preprint arXiv:2509.23437},
  year={2025}
}
```
