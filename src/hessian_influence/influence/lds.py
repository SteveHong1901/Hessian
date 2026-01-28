from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import bootstrap, spearmanr

from hessian_influence.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LDSResult:
    mean: float
    confidence_interval: float


class LDSEvaluator:
    def __init__(
        self,
        lso_path: Path,
        num_masks: int = 100,
        ind_repeat: int = 100,
        random_state: int = 42,
    ) -> None:
        self._lso_path = Path(lso_path)
        self._num_masks = num_masks
        self._ind_repeat = ind_repeat
        self._random_state = random_state

    def evaluate(
        self,
        data_name: str,
        influence_scores: torch.Tensor,
        alpha_values: list[float],
    ) -> dict[float, LDSResult]:
        results: dict[float, LDSResult] = {}

        for alpha in alpha_values:
            score_file = (
                self._lso_path
                / f"data_{data_name}"
                / f"{alpha}_{self._ind_repeat}r_1br.pt"
            )
            lso_data = torch.load(score_file, map_location="cpu", weights_only=False)

            diff_scores = lso_data["loss_diffs"][: self._num_masks, :]
            diff_scores = self._squeeze_if_needed(diff_scores)

            binary_masks = torch.from_numpy(lso_data["binary_masks"]).T.to(
                dtype=influence_scores.dtype
            )

            projected_scores = (influence_scores @ binary_masks).t()
            lds_result = self._compute_lds(diff_scores, projected_scores)

            results[alpha] = lds_result
            logger.debug(f"LDS for alpha={alpha}: mean={lds_result.mean:.4f}")

        return results

    def _compute_lds(
        self,
        ground_truth: torch.Tensor,
        predictions: torch.Tensor,
        max_valid: Optional[int] = None,
    ) -> LDSResult:
        num_masks, num_valid = ground_truth.shape
        if max_valid is not None:
            num_valid = max_valid

        def lds_statistic(mask_indices: tuple) -> float:
            correlations = []
            for i in range(num_valid):
                corr, _ = spearmanr(
                    ground_truth[list(mask_indices), i].numpy(),
                    predictions[list(mask_indices), i].numpy(),
                )
                correlations.append(corr)
            return float(np.nanmean(correlations))

        data = (tuple(range(num_masks)),)
        bootstrap_result = bootstrap(
            data,
            lds_statistic,
            n_resamples=num_masks,
            batch=num_masks * 2,
            random_state=self._random_state,
        )

        ci_low = bootstrap_result.confidence_interval[0]
        ci_high = bootstrap_result.confidence_interval[1]
        mean = (ci_low + ci_high) / 2
        ci_width = mean - ci_low

        return LDSResult(mean=float(mean), confidence_interval=float(ci_width))

    @staticmethod
    def _squeeze_if_needed(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(tensor, "dim"):
            if tensor.dim() > 2:
                return tensor.squeeze(-1)
        elif hasattr(tensor, "ndim"):
            if tensor.ndim > 2:
                return tensor.squeeze(-1)
        return tensor
