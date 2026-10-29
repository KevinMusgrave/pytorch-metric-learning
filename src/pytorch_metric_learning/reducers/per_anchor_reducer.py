from collections.abc import Callable

import torch
from torch import Tensor

from ..types import LossDict
from ..utils._common import arange_like, pos_inf
from .base_reducer import BaseReducer
from .mean_reducer import MeanReducer


def mean_per_anchor(x: Tensor, num_per_row: Tensor) -> Tensor:
    return torch.where(num_per_row == 0, 0, x.sum(dim=1) / num_per_row.clamp_min(1))


class PerAnchorReducer(BaseReducer):
    def __init__(
        self,
        reducer: BaseReducer | None = None,
        aggregation_func: Callable[[Tensor, Tensor], Tensor] = mean_per_anchor,
    ) -> None:
        super().__init__()
        self.reducer = reducer if reducer is not None else MeanReducer()
        self.aggregation_func = aggregation_func

    def element_reduction(
        self,
        losses: Tensor,
        loss_indices: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        loss_dict: LossDict = {
            "loss": {
                "losses": losses,
                "indices": loss_indices,
                "reduction_type": "element",
            }
        }
        return self.reducer(loss_dict, embeddings, labels)

    def pair_reduction(
        self,
        losses: Tensor,
        loss_indices: tuple[Tensor, Tensor],
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        batch_size = embeddings.shape[0]
        anchors, others = loss_indices
        new_array = torch.full(
            (batch_size, batch_size),
            pos_inf(losses.dtype),
            device=losses.device,
            dtype=losses.dtype,
        )
        new_array[anchors, others] = losses
        pos_inf_mask = new_array == pos_inf(losses.dtype)
        num_per_row = batch_size - torch.sum(pos_inf_mask, dim=1)
        new_array = torch.where(pos_inf_mask, torch.zeros_like(new_array), new_array)
        output = self.aggregation_func(new_array, num_per_row)
        return self.element_reduction(
            output, arange_like(embeddings), embeddings, labels
        )

    def pos_pair_reduction(
        self,
        losses: Tensor,
        loss_indices: tuple[Tensor, Tensor],
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        return self.pair_reduction(losses, loss_indices, embeddings, labels)

    def neg_pair_reduction(
        self,
        losses: Tensor,
        loss_indices: tuple[Tensor, Tensor],
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        return self.pair_reduction(losses, loss_indices, embeddings, labels)

    def triplet_reduction(
        self,
        losses: Tensor,
        loss_indices: tuple[Tensor, Tensor, Tensor],
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        raise NotImplementedError("PerAnchorReducer does not support triplets")
