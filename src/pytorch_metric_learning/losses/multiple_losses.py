from collections.abc import Mapping

import torch
from torch import Tensor

from ..types import IndicesTuple


class MultipleLosses(torch.nn.Module):
    def __init__(
        self,
        losses: Mapping[str, torch.nn.Module],
        miners: Mapping[str, torch.nn.Module] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(losses, Mapping) or not losses:
            raise TypeError("losses must be a non-empty mapping from names to modules")
        self.losses = torch.nn.ModuleDict(losses)
        miners = {} if miners is None else dict(miners)
        if not set(miners) <= set(losses):
            raise ValueError("miner keys must be present in losses")
        self.miners = torch.nn.ModuleDict(miners)
        self.weights = dict.fromkeys(losses, 1.0)
        if weights is not None:
            if set(weights) != set(losses):
                raise ValueError("weights must have exactly the same keys as losses")
            self.weights = dict(weights)

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: IndicesTuple | None = None,
    ) -> Tensor:
        if self.miners and indices_tuple is not None:
            raise ValueError("indices_tuple cannot be supplied with configured miners")
        total = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
        for name, loss_func in self.losses.items():
            current = indices_tuple
            if name in self.miners:
                current = self.miners[name](embeddings, labels)
            total = total + loss_func(embeddings, labels, current) * self.weights[name]
        return total
