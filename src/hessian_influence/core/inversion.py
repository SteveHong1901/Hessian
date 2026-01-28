from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

import numpy as np
import torch
from curvlinops import KFACInverseLinearOperator
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from hessian_influence.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from curvlinops._torch_base import PyTorchLinearOperator
except ImportError:
    PyTorchLinearOperator = ()


class InversionMethod(str, Enum):
    EXACT = "exact"
    PSEUDO = "pseudo"
    EIGEN = "eigen"
    KFAC = "kfac"


ReplaceStrategy = Literal["threshold", "positive", "keep_negative"]


@dataclass
class InversionConfig:
    method: InversionMethod = InversionMethod.EIGEN
    damping: float = 0.0
    eps: float = 1e-6
    rcond: float = 1e-6
    replace_with: Union[float, ReplaceStrategy] = 0.0


class MatrixInverter:
    def __init__(self, config: Optional[InversionConfig] = None) -> None:
        self._config = config or InversionConfig()
        logger.info(f"Initialized MatrixInverter with method={self._config.method.value}")

    @staticmethod
    def to_numpy(matrix: object) -> np.ndarray:
        if isinstance(matrix, torch.Tensor):
            return matrix.detach().cpu().numpy()

        if isinstance(matrix, PyTorchLinearOperator):
            dim = matrix.shape[0]
            identity = torch.eye(dim, dtype=torch.float32, device="cpu")
            dense = matrix @ identity
            return dense.detach().cpu().numpy()

        if isinstance(matrix, LinearOperator) or hasattr(matrix, "matmat"):
            dim = matrix.shape[0]
            identity = np.eye(dim, dtype=float)
            return matrix @ identity

        return np.asarray(matrix)

    def damp(self, matrix: object, damping: Optional[float] = None) -> np.ndarray:
        damp_val = damping if damping is not None else self._config.damping
        mat = self.to_numpy(matrix)

        if damp_val == 0.0:
            return mat

        dim = mat.shape[0]
        return mat + damp_val * np.eye(dim, dtype=mat.dtype)

    def invert(
        self,
        matrix: object,
        method: Optional[InversionMethod] = None,
        damping: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        inv_method = method or self._config.method
        logger.debug(f"Inverting matrix using {inv_method.value} method")

        if inv_method == InversionMethod.EXACT:
            return self._exact_inverse(matrix, damping)
        elif inv_method == InversionMethod.PSEUDO:
            rcond = kwargs.get("rcond", self._config.rcond)
            return self._pseudo_inverse(matrix, damping, rcond)
        elif inv_method == InversionMethod.EIGEN:
            eps = kwargs.get("eps", self._config.eps)
            replace_with = kwargs.get("replace_with", self._config.replace_with)
            return self._eigen_inverse(matrix, damping, eps, replace_with)
        else:
            raise ValueError(f"Unsupported inversion method: {inv_method}")

    def _exact_inverse(
        self,
        matrix: object,
        damping: Optional[float] = None,
    ) -> np.ndarray:
        mat = self.damp(matrix, damping)
        return np.linalg.inv(mat)

    def _pseudo_inverse(
        self,
        matrix: object,
        damping: Optional[float] = None,
        rcond: Optional[float] = None,
    ) -> np.ndarray:
        mat = self.damp(matrix, damping)
        cutoff = rcond or self._config.rcond

        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        if s.size == 0:
            return np.zeros_like(mat.T)

        threshold = cutoff * s[0]
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        return (Vt.T * s_inv) @ U.T

    def _eigen_inverse(
        self,
        matrix: object,
        damping: Optional[float] = None,
        eps: Optional[float] = None,
        replace_with: Optional[Union[float, ReplaceStrategy]] = None,
    ) -> np.ndarray:
        mat = self.damp(matrix, damping)
        epsilon = eps if eps is not None else self._config.eps
        replacement = replace_with if replace_with is not None else self._config.replace_with

        eigenvalues, eigenvectors = np.linalg.eigh(mat)

        pos_mask = eigenvalues > epsilon
        neg_mask = eigenvalues < -epsilon

        if isinstance(replacement, str) and replacement == "keep_negative":
            keep_mask = pos_mask | neg_mask
            if not np.any(keep_mask):
                return np.zeros_like(mat)

            inv_eigenvalues = np.zeros_like(eigenvalues)
            inv_eigenvalues[keep_mask] = 1.0 / eigenvalues[keep_mask]
            return eigenvectors @ np.diag(inv_eigenvalues) @ eigenvectors.T

        if not np.any(pos_mask):
            if not (isinstance(replacement, str) and replacement == "positive" and np.any(neg_mask)):
                return np.zeros_like(mat)

        inv_eigenvalues = np.empty_like(eigenvalues)

        if isinstance(replacement, str):
            if replacement == "threshold":
                inv_eigenvalues[pos_mask] = 1.0 / eigenvalues[pos_mask]
                inv_eigenvalues[~pos_mask] = epsilon
            elif replacement == "positive":
                inv_eigenvalues[pos_mask] = 1.0 / eigenvalues[pos_mask]
                inv_eigenvalues[neg_mask] = 1.0 / (-eigenvalues[neg_mask])
                between_mask = (~pos_mask) & (~neg_mask)
                inv_eigenvalues[between_mask] = epsilon
            else:
                raise ValueError(f"Unknown replacement strategy: {replacement}")
        else:
            inv_eigenvalues[pos_mask] = 1.0 / eigenvalues[pos_mask]
            inv_eigenvalues[~pos_mask] = replacement

        return eigenvectors @ np.diag(inv_eigenvalues) @ eigenvectors.T

    def invert_with_info(
        self,
        matrix: object,
        damping: Optional[float] = None,
        eps: Optional[float] = None,
        replace_with: Optional[Union[float, ReplaceStrategy]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mat = self.damp(matrix, damping)
        epsilon = eps if eps is not None else self._config.eps
        replacement = replace_with if replace_with is not None else self._config.replace_with

        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        inverse = self._eigen_inverse(matrix, damping, epsilon, replacement)

        return inverse, mat, eigenvalues, eigenvectors

    def invert_block_diagonal(
        self,
        matrix: object,
        block_sizes: list[int],
        damping: Optional[float] = None,
        eps: Optional[float] = None,
        replace_with: Optional[Union[float, ReplaceStrategy]] = None,
    ) -> np.ndarray:
        mat = self.to_numpy(matrix)
        epsilon = eps if eps is not None else self._config.eps
        replacement = replace_with if replace_with is not None else self._config.replace_with

        logger.debug(f"Inverting block diagonal with {len(block_sizes)} blocks")

        inv_blocks = []
        start = 0
        for size in block_sizes:
            block = mat[start : start + size, start : start + size]
            inv_block = self._eigen_inverse(block, damping, epsilon, replacement)
            inv_blocks.append(inv_block)
            start += size

        return sparse.block_diag(inv_blocks).toarray()

    @staticmethod
    def invert_kfac(
        kfac_operator: object,
        damping: float = 1e-6,
        use_ekfac: bool = False,
        use_heuristic_damping: bool = True,
    ) -> KFACInverseLinearOperator:
        logger.debug(f"Computing KFAC inverse with damping={damping}")

        if use_ekfac:
            return KFACInverseLinearOperator(
                A=kfac_operator,
                use_exact_damping=True,
                damping=damping,
            )

        return KFACInverseLinearOperator(
            A=kfac_operator,
            damping=damping,
            use_heuristic_damping=use_heuristic_damping,
        )
