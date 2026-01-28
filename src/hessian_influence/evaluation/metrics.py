from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator, eigsh

from hessian_influence.core.inversion import MatrixInverter
from hessian_influence.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StabilityStats:
    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    num_positive: int
    num_negative: int
    num_small_positive: int


@dataclass
class EigenDecomposition:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    figure: Optional[plt.Figure] = None


class HessianMetrics:
    def __init__(self) -> None:
        self._inverter = MatrixInverter()

    def frobenius_norm(
        self,
        matrix_a: object,
        matrix_b: Optional[object] = None,
    ) -> float:
        a_np = self._inverter.to_numpy(matrix_a)

        if matrix_b is None:
            return float(np.linalg.norm(a_np, ord="fro"))

        b_np = self._inverter.to_numpy(matrix_b)
        return float(np.linalg.norm(a_np - b_np, ord="fro"))

    def relative_residual(self, target: object, approximation: object) -> float:
        target_norm = self.frobenius_norm(target)
        if target_norm == 0:
            return float("nan")
        return float(self.frobenius_norm(target, approximation) / target_norm)

    def off_block_energy_ratio(self, matrix: object, block_sizes: list[int]) -> float:
        mat = self._inverter.to_numpy(matrix)
        block_diag = np.zeros_like(mat)

        start = 0
        for size in block_sizes:
            block_diag[start : start + size, start : start + size] = mat[
                start : start + size, start : start + size
            ]
            start += size

        diff_norm = np.linalg.norm(mat - block_diag, ord="fro")
        total_norm = np.linalg.norm(mat, ord="fro")

        if total_norm == 0:
            return float("nan")
        return float(diff_norm / total_norm)

    def eigenvalue_overlap(
        self,
        approximation: object,
        reference: object,
        block_sizes: list[int],
        k: Optional[int] = None,
    ) -> list[float]:
        approx = self._inverter.to_numpy(approximation)
        ref = self._inverter.to_numpy(reference)

        overlaps: list[float] = []
        start = 0

        for size in block_sizes:
            end = start + size
            k_block = min(k, size) if k is not None else None

            eig_approx = self._compute_eigenvalues(
                approx[start:end, start:end], k=k_block
            )
            eig_ref = self._compute_eigenvalues(
                ref[start:end, start:end], k=k_block
            )

            eig_approx = np.sort(eig_approx)
            eig_ref = np.sort(eig_ref)

            n = min(len(eig_approx), len(eig_ref))
            diff = eig_approx[:n] - eig_ref[:n]
            denom = np.linalg.norm(eig_ref[:n], ord=2)

            if denom == 0:
                overlaps.append(float("nan"))
            else:
                overlaps.append(float(1.0 - np.linalg.norm(diff, ord=2) / denom))

            start = end

        return overlaps

    def eigenbasis_overlap(
        self,
        approximation: object,
        reference: object,
        block_sizes: list[int],
        k: int,
    ) -> list[float]:
        approx = self._inverter.to_numpy(approximation)
        ref = self._inverter.to_numpy(reference)

        overlaps: list[float] = []
        start = 0

        for size in block_sizes:
            end = start + size
            k_block = min(k, size)

            _, u_k = self._compute_eigen(approx[start:end, start:end], k=k_block)
            _, v_k = self._compute_eigen(ref[start:end, start:end], k=k_block)

            u_k = u_k[:, :k_block]
            v_k = v_k[:, :k_block]

            overlap = np.linalg.norm(v_k.T @ u_k, ord="fro") ** 2 / k_block
            overlaps.append(float(overlap))
            start = end

        return overlaps

    def eigen_decomposition(
        self,
        operator: object,
        k: Optional[int] = None,
        plot: bool = False,
    ) -> EigenDecomposition:
        if isinstance(operator, LinearOperator):
            if k is None or k >= operator.shape[0]:
                mat = self._inverter.to_numpy(operator)
                eigenvalues, eigenvectors = np.linalg.eigh(mat)
            else:
                eigenvalues, eigenvectors = eigsh(operator, k=k, which="LM")
        else:
            mat = self._inverter.to_numpy(operator)
            n = mat.shape[0]
            if k is None or k >= n:
                eigenvalues, eigenvectors = np.linalg.eigh(mat)
            else:
                eigenvalues, eigenvectors = eigsh(aslinearoperator(mat), k=k, which="LM")

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        figure = None
        if plot:
            figure, ax = plt.subplots()
            ax.semilogy(range(1, len(eigenvalues) + 1), np.abs(eigenvalues), marker="o")
            ax.set_xlabel("Eigenvalue Index")
            ax.set_ylabel("Magnitude (log scale)")
            ax.grid(True)

        return EigenDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            figure=figure,
        )

    def condition_number(self, operator: object) -> float:
        decomposition = self.eigen_decomposition(operator)
        eigenvalues = decomposition.eigenvalues

        if len(eigenvalues) == 0:
            return float("nan")

        return float(np.abs(eigenvalues[0]) / (np.abs(eigenvalues[-1]) + 1e-12))

    def stability_statistics(
        self,
        operator: object,
        small_threshold: float = 1e-6,
        verbose: bool = False,
    ) -> StabilityStats:
        decomposition = self.eigen_decomposition(operator)
        eigenvalues = decomposition.eigenvalues

        stats = StabilityStats(
            condition_number=(
                float(np.abs(eigenvalues[0]) / (np.abs(eigenvalues[-1]) + 1e-12))
                if eigenvalues.size
                else float("nan")
            ),
            min_eigenvalue=float(eigenvalues.min()) if eigenvalues.size else float("nan"),
            max_eigenvalue=float(eigenvalues.max()) if eigenvalues.size else float("nan"),
            num_positive=int(np.sum(eigenvalues > 0)),
            num_negative=int(np.sum(eigenvalues < 0)),
            num_small_positive=int(np.sum((eigenvalues > 0) & (eigenvalues < small_threshold))),
        )

        if verbose:
            logger.info(f"Condition number: {stats.condition_number:.2e}")
            logger.info(f"Min eigenvalue: {stats.min_eigenvalue:.2e}")
            logger.info(f"Max eigenvalue: {stats.max_eigenvalue:.2e}")
            logger.info(f"Positive eigenvalues: {stats.num_positive}")
            logger.info(f"Negative eigenvalues: {stats.num_negative}")
            logger.info(f"Small positive eigenvalues: {stats.num_small_positive}")

        return stats

    def _compute_eigenvalues(
        self,
        matrix: np.ndarray,
        k: Optional[int] = None,
    ) -> np.ndarray:
        if k is None or k >= matrix.shape[0]:
            eigenvalues, _ = np.linalg.eigh(matrix)
        else:
            eigenvalues, _ = eigsh(aslinearoperator(matrix), k=k, which="LM")
        return eigenvalues

    def _compute_eigen(
        self,
        matrix: np.ndarray,
        k: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if k is None or k >= matrix.shape[0]:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        else:
            eigenvalues, eigenvectors = eigsh(aslinearoperator(matrix), k=k, which="LM")

        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]
