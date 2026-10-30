from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from ..types import MetricLoss
from .base_metric_loss_function import BaseMetricLossFunction


class MultipleLosses(torch.nn.Module):
    def __init__(
        self,
        losses: Mapping[str, MetricLoss],
        miners: Mapping[str, torch.nn.Module] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(losses, Mapping) or not losses:
            raise TypeError("losses must be a non-empty mapping from names to modules")
        for loss in losses.values():
            _require_metric_loss(loss, "MultipleLosses")
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
        labels: Tensor | None = None,
        indices_tuple: Sequence[Tensor] | None = None,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
    ) -> Tensor:
        total = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
        for name, loss_func in self.losses.items():
            current = indices_tuple
            if name in self.miners:
                current = self.miners[name](
                    embeddings, labels, indices_tuple, ref_emb, ref_labels
                )
            loss = loss_func(embeddings, labels, current, ref_emb, ref_labels)
            total = total + loss * self.weights[name]
        return total


def _require_metric_loss(loss: object, owner: str) -> None:
    if not isinstance(loss, (BaseMetricLossFunction, MultipleLosses)):
        raise TypeError(
            f"{owner} requires a BaseMetricLossFunction or MultipleLosses, "
            f"not {type(loss).__name__}"
        )
