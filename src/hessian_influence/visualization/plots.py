from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from matplotlib.colors import LogNorm

from hessian_influence.core.inversion import MatrixInverter


class HessianVisualizer:
    def __init__(self) -> None:
        self._inverter = MatrixInverter()

    def plot_matrix(
        self,
        matrix: object,
        model: nn.Module,
        title: str = "Matrix",
        cmap: str = "viridis",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        layer_info = self._compute_layer_info(model)
        permutation = self._build_permutation(layer_info)
        mat_dense = self._inverter.to_numpy(matrix)
        mat_permuted = mat_dense[permutation][:, permutation]

        fig, ax = plt.subplots(figsize=(6, 6))

        mat_abs = np.clip(np.abs(mat_permuted), 1e-10, 1.0)
        im = ax.imshow(mat_abs, cmap=cmap, norm=LogNorm(vmin=1e-10, vmax=1.0))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)

        self._draw_layer_boundaries(ax, layer_info)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_matrix_comparison(
        self,
        matrix1: object,
        matrix2: object,
        model: nn.Module,
        title1: str = "Matrix 1",
        title2: str = "Matrix 2",
        title_diff: str = "Difference",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        layer_info = self._compute_layer_info(model)
        permutation = self._build_permutation(layer_info)

        mat1 = self._inverter.to_numpy(matrix1)
        mat2 = self._inverter.to_numpy(matrix2)

        mat1_permuted = mat1[permutation][:, permutation]
        mat2_permuted = mat2[permutation][:, permutation]
        diff_permuted = mat1_permuted - mat2_permuted

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        matrices = [mat1_permuted, mat2_permuted, diff_permuted]
        titles = [title1, title2, title_diff]

        for ax, mat, title in zip(axes, matrices, titles):
            im = ax.imshow(
                np.log10(np.abs(mat) + 1e-12),
                cmap="viridis",
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title)
            self._draw_layer_boundaries(ax, layer_info)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_lds_scores(
        self,
        results: dict[str, dict[float, dict]],
        xlabel: str = "Alpha",
        ylabel: str = "LDS Mean",
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots()

        for label, series in results.items():
            alphas = sorted(series.keys())
            means = [series[a]["mean"] for a in alphas]
            cis = [series[a]["ci"] for a in alphas]

            ax.errorbar(
                alphas,
                means,
                yerr=cis,
                marker="o",
                linestyle="-",
                label=label,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_eigenvalue_spectrum(
        self,
        eigenvalues: np.ndarray,
        title: str = "Eigenvalue Spectrum",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots()

        ax.semilogy(
            range(1, len(eigenvalues) + 1),
            np.abs(eigenvalues),
            marker="o",
            markersize=3,
        )
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Magnitude (log scale)")
        ax.set_title(title)
        ax.grid(True)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def _compute_layer_info(
        self,
        model: nn.Module,
    ) -> dict[str, list[tuple[int, int]]]:
        layer_slices: dict[str, list[tuple[int, int]]] = {}
        idx = 0

        for full_name, param in model.named_parameters():
            numel = param.numel()
            layer = full_name.rsplit(".", 1)[0]

            if layer not in layer_slices:
                layer_slices[layer] = []
            layer_slices[layer].append((idx, idx + numel))
            idx += numel

        return layer_slices

    def _build_permutation(
        self,
        layer_info: dict[str, list[tuple[int, int]]],
    ) -> np.ndarray:
        permutation = []
        for layer in layer_info:
            for start, end in layer_info[layer]:
                permutation.extend(range(start, end))
        return np.array(permutation, dtype=int)

    def _draw_layer_boundaries(
        self,
        ax: plt.Axes,
        layer_info: dict[str, list[tuple[int, int]]],
    ) -> None:
        block_sizes = [
            sum(end - start for start, end in slices)
            for slices in layer_info.values()
        ]
        cumulative = np.cumsum(block_sizes)

        for pos in cumulative[:-1]:
            ax.axhline(pos - 0.5, color="white", linewidth=1)
            ax.axvline(pos - 0.5, color="white", linewidth=1)
