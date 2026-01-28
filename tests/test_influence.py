import numpy as np
import pytest
import torch

from hessian_influence.influence.calculator import InfluenceCalculator


@pytest.fixture
def calculator() -> InfluenceCalculator:
    return InfluenceCalculator(device="cpu")


class TestInfluenceCalculator:
    def test_compute_influence_scores(self, calculator: InfluenceCalculator) -> None:
        test_grad = torch.tensor([1.0, 2.0])
        train_grads = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        inverse_hessian = torch.eye(2)

        scores = calculator.compute_influence_scores(
            test_grad, train_grads, inverse_hessian
        )

        assert scores.shape == (3,)
        expected = -train_grads.double() @ test_grad.double()
        assert torch.allclose(scores, expected)

    def test_compute_influence_scores_batch(
        self, calculator: InfluenceCalculator
    ) -> None:
        test_grads = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        train_grads = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        inverse_hessian = torch.eye(2)

        scores = calculator.compute_influence_scores_batch(
            test_grads, train_grads, inverse_hessian
        )

        assert scores.shape == (2, 2)

    def test_compute_influence_scores_with_list(
        self, calculator: InfluenceCalculator
    ) -> None:
        test_grad = torch.tensor([1.0, 2.0])
        train_grads = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        inverse_hessian = torch.eye(2)

        scores = calculator.compute_influence_scores(
            test_grad, train_grads, inverse_hessian
        )

        assert scores.shape == (2,)
