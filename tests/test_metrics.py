import numpy as np
import pytest

from hessian_influence.evaluation.metrics import HessianMetrics


@pytest.fixture
def metrics() -> HessianMetrics:
    return HessianMetrics()


class TestHessianMetrics:
    def test_frobenius_norm_single_matrix(self, metrics: HessianMetrics) -> None:
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        expected = np.sqrt(2.0)
        assert np.isclose(metrics.frobenius_norm(matrix), expected)

    def test_frobenius_norm_difference(self, metrics: HessianMetrics) -> None:
        matrix_a = np.eye(2)
        matrix_b = np.diag([1.0, 2.0])
        expected = np.linalg.norm(matrix_a - matrix_b, ord="fro")
        assert np.isclose(metrics.frobenius_norm(matrix_a, matrix_b), expected)

    def test_relative_residual(self, metrics: HessianMetrics) -> None:
        target = np.eye(2)
        approximation = np.diag([1.0, 2.0])
        expected = np.linalg.norm(target - approximation, ord="fro") / np.linalg.norm(
            target, ord="fro"
        )
        assert np.isclose(metrics.relative_residual(target, approximation), expected)

    def test_off_block_energy_ratio(self, metrics: HessianMetrics) -> None:
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.linalg.norm(
            matrix - np.diag(np.diag(matrix)), ord="fro"
        ) / np.linalg.norm(matrix, ord="fro")
        assert np.isclose(metrics.off_block_energy_ratio(matrix, [1, 1]), expected)

    def test_eigenvalue_overlap(self, metrics: HessianMetrics) -> None:
        approx = np.diag([1.0, 2.0, 4.0, 5.0])
        reference = np.diag([1.0, 3.0, 4.0, 6.0])

        exp1 = 1.0 - np.linalg.norm(
            np.sort([1.0, 2.0]) - np.sort([1.0, 3.0])
        ) / np.linalg.norm(np.sort([1.0, 3.0]))
        exp2 = 1.0 - np.linalg.norm(
            np.sort([4.0, 5.0]) - np.sort([4.0, 6.0])
        ) / np.linalg.norm(np.sort([4.0, 6.0]))

        expected = [exp1, exp2]
        result = metrics.eigenvalue_overlap(approx, reference, block_sizes=[2, 2])
        assert np.allclose(result, expected)

    def test_eigenbasis_overlap(self, metrics: HessianMetrics) -> None:
        block1_approx = np.diag([1.0, 2.0])
        block2_approx = np.diag([3.0, 4.0])
        approx = np.block([
            [block1_approx, np.zeros((2, 2))],
            [np.zeros((2, 2)), block2_approx],
        ])

        block1_ref = np.array([[2.0, 1.0], [1.0, 2.0]])
        block2_ref = np.diag([4.0, 5.0])
        reference = np.block([
            [block1_ref, np.zeros((2, 2))],
            [np.zeros((2, 2)), block2_ref],
        ])

        expected = [0.5, 1.0]
        result = metrics.eigenbasis_overlap(approx, reference, [2, 2], k=1)
        assert np.allclose(result, expected)

    def test_condition_number(self, metrics: HessianMetrics) -> None:
        matrix = np.diag([1.0, 100.0])
        assert np.isclose(metrics.condition_number(matrix), 100.0)

    def test_eigen_decomposition(self, metrics: HessianMetrics) -> None:
        matrix = np.diag([3.0, 1.0, 2.0])
        result = metrics.eigen_decomposition(matrix)

        assert len(result.eigenvalues) == 3
        assert np.allclose(np.sort(result.eigenvalues)[::-1], [3.0, 2.0, 1.0])
        assert result.eigenvectors.shape == (3, 3)
