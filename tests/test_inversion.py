import numpy as np
import pytest

from hessian_influence.core.inversion import (
    InversionConfig,
    InversionMethod,
    MatrixInverter,
)


@pytest.fixture
def inverter() -> MatrixInverter:
    return MatrixInverter()


class TestMatrixInverter:
    def test_exact_inverse(self, inverter: MatrixInverter) -> None:
        matrix = np.array([[2.0, 0.0], [0.0, 4.0]])
        result = inverter.invert(matrix, method=InversionMethod.EXACT)
        expected = np.array([[0.5, 0.0], [0.0, 0.25]])
        assert np.allclose(result, expected)

    def test_pseudo_inverse(self, inverter: MatrixInverter) -> None:
        matrix = np.array([[2.0, 0.0], [0.0, 4.0]])
        result = inverter.invert(matrix, method=InversionMethod.PSEUDO)
        expected = np.array([[0.5, 0.0], [0.0, 0.25]])
        assert np.allclose(result, expected)

    def test_eigen_inverse(self, inverter: MatrixInverter) -> None:
        matrix = np.array([[2.0, 0.0], [0.0, 4.0]])
        result = inverter.invert(matrix, method=InversionMethod.EIGEN)
        expected = np.array([[0.5, 0.0], [0.0, 0.25]])
        assert np.allclose(result, expected)

    def test_damping(self, inverter: MatrixInverter) -> None:
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        damped = inverter.damp(matrix, damping=1.0)
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        assert np.allclose(damped, expected)

    def test_eigen_inverse_with_threshold(self, inverter: MatrixInverter) -> None:
        matrix = np.array([[1.0, 0.0], [0.0, 1e-10]])
        result = inverter.invert(
            matrix,
            method=InversionMethod.EIGEN,
            eps=1e-6,
            replace_with=0.0,
        )
        expected = np.array([[1.0, 0.0], [0.0, 0.0]])
        assert np.allclose(result, expected)

    def test_block_diagonal_inverse(self, inverter: MatrixInverter) -> None:
        block1 = np.array([[2.0, 0.0], [0.0, 4.0]])
        block2 = np.array([[1.0, 0.0], [0.0, 2.0]])
        matrix = np.block([
            [block1, np.zeros((2, 2))],
            [np.zeros((2, 2)), block2],
        ])

        result = inverter.invert_block_diagonal(matrix, block_sizes=[2, 2])

        expected_block1 = np.array([[0.5, 0.0], [0.0, 0.25]])
        expected_block2 = np.array([[1.0, 0.0], [0.0, 0.5]])
        expected = np.block([
            [expected_block1, np.zeros((2, 2))],
            [np.zeros((2, 2)), expected_block2],
        ])

        assert np.allclose(result, expected)

    def test_invert_with_info(self, inverter: MatrixInverter) -> None:
        matrix = np.array([[2.0, 0.0], [0.0, 4.0]])
        inv, mat, eigenvalues, eigenvectors = inverter.invert_with_info(matrix)

        assert inv.shape == (2, 2)
        assert mat.shape == (2, 2)
        assert len(eigenvalues) == 2
        assert eigenvectors.shape == (2, 2)
