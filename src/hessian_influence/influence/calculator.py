from typing import Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hessian_influence.utils.logging import get_logger

logger = get_logger(__name__)


class InfluenceCalculator:
    def __init__(self, device: Union[str, torch.device] = "cpu") -> None:
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def compute_gradient(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        accumulated_grads = [torch.zeros_like(p, device=self._device) for p in params]

        total_samples = 0
        for inputs, targets in dataloader:
            inputs = inputs.view(inputs.size(0), -1).to(self._device)
            targets = targets.to(self._device)
            batch_size = inputs.size(0)

            model.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            (loss * batch_size).backward()

            for acc, p in zip(accumulated_grads, params):
                acc += p.grad.detach()

            total_samples += batch_size

        for acc in accumulated_grads:
            acc.div_(total_samples)

        flat_grad = torch.cat([g.contiguous().view(-1) for g in accumulated_grads])
        return flat_grad.unsqueeze(0)

    def compute_sample_gradients(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        single_loader = DataLoader(dataloader.dataset, batch_size=1, shuffle=False)
        gradients = []

        for x, y in single_loader:
            dataset = torch.utils.data.TensorDataset(
                x.to(self._device), y.to(self._device)
            )
            loader = DataLoader(dataset, batch_size=1)
            grad = self.compute_gradient(model, loader, loss_fn)
            gradients.append(grad.squeeze(0))

        return torch.stack(gradients)

    def compute_influence_scores(
        self,
        test_gradient: torch.Tensor,
        train_gradients: Union[torch.Tensor, list[torch.Tensor]],
        inverse_hessian: Any,
    ) -> torch.Tensor:
        test = test_gradient.view(-1)

        if isinstance(train_gradients, list):
            train = torch.stack([g.view(-1) for g in train_gradients])
        else:
            train = train_gradients.view(train_gradients.size(0), -1)

        is_linear_operator = hasattr(inverse_hessian, "_matmat") or hasattr(
            inverse_hessian, "matvec"
        )

        if is_linear_operator:
            test = test.float()
            train = train.float()
        else:
            test = test.double()
            train = train.double()
            inverse_hessian = inverse_hessian.double()

        v = inverse_hessian @ test
        scores = -(train @ v)

        return scores.squeeze()

    def compute_influence_scores_batch(
        self,
        test_gradients: torch.Tensor,
        train_gradients: torch.Tensor,
        inverse_hessian: Any,
    ) -> torch.Tensor:
        test = test_gradients.view(test_gradients.size(0), -1)
        train = train_gradients.view(train_gradients.size(0), -1)

        is_linear_operator = hasattr(inverse_hessian, "_matmat") or hasattr(
            inverse_hessian, "matvec"
        )

        if is_linear_operator:
            test = test.to(torch.float32)
            train = train.to(torch.float32)
        else:
            test = test.to(torch.float64)
            train = train.to(torch.float64)
            inverse_hessian = inverse_hessian.to(torch.float64)

        v = inverse_hessian @ test.T
        scores = -(train @ v).T

        return scores
