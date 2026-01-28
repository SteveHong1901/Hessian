from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from curvlinops import (
    EKFACLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
)
from curvlinops.diagonal.hutchinson import hutchinson_diag
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg

from hessian_influence.utils.logging import get_logger

logger = get_logger(__name__)


class CurvatureType(str, Enum):
    HESSIAN = "H"
    GGN = "GGN"
    KFAC = "KFAC"
    EKFAC = "EKFAC"


@dataclass
class HessianConfig:
    curvature_type: CurvatureType = CurvatureType.HESSIAN
    num_probes: int = 1000
    damping: float = 0.0
    cg_rtol: float = 1e-5
    cg_maxiter: Optional[int] = None


class HessianComputer:
    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        data: torch.utils.data.DataLoader,
        config: Optional[HessianConfig] = None,
    ) -> None:
        self._model = model
        self._loss_function = loss_function
        self._data = data
        self._config = config or HessianConfig()
        self._params: list[torch.Tensor] = [p for p in model.parameters() if p.requires_grad]
        self._cached_operator: Optional[LinearOperator] = None
        logger.info(f"Initialized HessianComputer with {self._config.curvature_type.value}")

    @property
    def params(self) -> list[torch.Tensor]:
        return self._params

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self._params)

    def compute_operator(self, curvature_type: Optional[CurvatureType] = None) -> LinearOperator:
        ctype = curvature_type or self._config.curvature_type
        logger.debug(f"Computing {ctype.value} operator")

        if ctype == CurvatureType.HESSIAN:
            operator = HessianLinearOperator(
                self._model, self._loss_function, self._params, self._data
            ).to_scipy()
        elif ctype == CurvatureType.GGN:
            operator = GGNLinearOperator(
                self._model, self._loss_function, self._params, self._data
            ).to_scipy()
        else:
            raise ValueError(f"Unsupported curvature type: {ctype}")

        self._cached_operator = operator
        return operator

    def compute_kfac_operator(
        self,
        num_data: Optional[int] = None,
        use_ekfac: bool = False,
    ) -> KFACLinearOperator | EKFACLinearOperator:
        if num_data is None:
            try:
                num_data = len(self._data.dataset)
            except AttributeError:
                raise ValueError("num_data must be provided if data has no .dataset attribute")

        logger.debug(f"Computing {'EKFAC' if use_ekfac else 'KFAC'} operator")

        operator_cls = EKFACLinearOperator if use_ekfac else KFACLinearOperator
        return operator_cls(
            model_func=self._model,
            loss_func=self._loss_function,
            params=self._params,
            data=self._data,
            num_data=num_data,
        )

    def compute_diagonal(
        self,
        curvature_type: Optional[CurvatureType] = None,
        num_probes: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        ctype = curvature_type or self._config.curvature_type
        n_probes = num_probes or self._config.num_probes

        operator = self.compute_operator(ctype)
        dim = operator.shape[0]
        n_probes = min(n_probes, dim - 1)

        logger.debug(f"Computing diagonal with {n_probes} probes")
        diag_values = hutchinson_diag(operator, n_probes)

        diag_tensor = torch.from_numpy(diag_values)
        if device is not None:
            diag_tensor = diag_tensor.to(device)

        return torch.diag(diag_tensor)

    def compute_cg_inverse_operator(
        self,
        curvature_type: Optional[CurvatureType] = None,
        damping: Optional[float] = None,
        rtol: Optional[float] = None,
        maxiter: Optional[int] = None,
    ) -> LinearOperator:
        ctype = curvature_type or self._config.curvature_type
        damp = damping if damping is not None else self._config.damping
        tol = rtol or self._config.cg_rtol
        max_it = maxiter or self._config.cg_maxiter

        base_operator = self.compute_operator(ctype)
        damping_operator = aslinearoperator(damp * sparse.eye(base_operator.shape[0]))
        damped_operator = base_operator + damping_operator

        logger.debug(f"Creating CG inverse operator with damping={damp}, rtol={tol}")

        def matvec(x: np.ndarray) -> np.ndarray:
            result, _ = cg(damped_operator, x, maxiter=max_it, rtol=tol)
            return result

        def matmat(X: np.ndarray) -> np.ndarray:
            return np.column_stack([matvec(col) for col in X.T])

        return LinearOperator(
            damped_operator.shape,
            matvec=matvec,
            matmat=matmat,
            dtype=damped_operator.dtype,
        )

    def compute_block_diagonal(
        self,
        block_sizes: Sequence[int],
        curvature_type: Optional[CurvatureType] = None,
    ) -> LinearOperator:
        ctype = curvature_type or self._config.curvature_type
        logger.debug(f"Computing block diagonal {ctype.value} with {len(block_sizes)} blocks")

        if ctype == CurvatureType.HESSIAN:
            return HessianLinearOperator(
                self._model,
                self._loss_function,
                self._params,
                self._data,
                block_sizes=block_sizes,
            ).to_scipy()

        if ctype == CurvatureType.GGN:
            base = GGNLinearOperator(
                self._model, self._loss_function, self._params, self._data
            ).to_scipy()
            mat = self._to_dense(base)

            blocks = []
            start = 0
            for size in block_sizes:
                block = mat[start : start + size, start : start + size]
                blocks.append(block)
                start += size

            return aslinearoperator(sparse.block_diag(blocks))

        raise ValueError(f"Block diagonal not supported for {ctype}")

    @staticmethod
    def _to_dense(operator: LinearOperator) -> np.ndarray:
        dim = operator.shape[0]
        identity = np.eye(dim, dtype=float)
        return operator @ identity
