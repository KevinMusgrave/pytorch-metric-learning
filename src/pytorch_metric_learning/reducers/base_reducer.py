import functools
import operator
from typing import Any

import torch
from torch import Tensor

from ..types import LossDict, LossTerm
from ..utils._common import zero_loss


class BaseReducer(torch.nn.Module):
    def forward(
        self, loss_dict: LossDict, embeddings: Tensor, labels: Tensor | None
    ) -> Tensor:
        reduced = [self.reduce(term, embeddings, labels) for term in loss_dict.values()]
        return functools.reduce(operator.add, reduced)

    def reduce(
        self, term: LossTerm, embeddings: Tensor, labels: Tensor | None
    ) -> Tensor:
        losses, indices = validate_term(term)
        reduction_type = term["reduction_type"]
        if reduction_type == "already_reduced":
            return losses
        if losses.numel() == 0:
            return zero_loss(embeddings)
        if reduction_type == "element":
            return self.element_reduction(losses, indices, embeddings, labels)
        if reduction_type == "pos_pair":
            return self.pos_pair_reduction(losses, indices, embeddings, labels)
        if reduction_type == "neg_pair":
            return self.neg_pair_reduction(losses, indices, embeddings, labels)
        return self.triplet_reduction(losses, indices, embeddings, labels)

    def element_reduction(
        self,
        losses: Tensor,
        loss_indices: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        raise NotImplementedError

    def pos_pair_reduction(
        self,
        losses: Tensor,
        loss_indices: tuple[Tensor, Tensor],
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        return self.element_reduction(losses, loss_indices[0], embeddings, labels)

    def neg_pair_reduction(
        self,
        losses: Tensor,
        loss_indices: tuple[Tensor, Tensor],
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        return self.element_reduction(losses, loss_indices[0], embeddings, labels)

    def triplet_reduction(
        self,
        losses: Tensor,
        loss_indices: tuple[Tensor, Tensor, Tensor],
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        return self.element_reduction(losses, loss_indices[0], embeddings, labels)


INDEX_COUNTS = {"element": 1, "pos_pair": 2, "neg_pair": 2, "triplet": 3}


def validate_term(term: LossTerm) -> tuple[Tensor, Any]:
    losses, indices, reduction_type = (
        term["losses"],
        term["indices"],
        term["reduction_type"],
    )
    if not isinstance(losses, Tensor):
        raise TypeError("losses must be a tensor")
    if reduction_type == "already_reduced":
        if losses.numel() != 1:
            raise ValueError("An already_reduced loss must contain one value")
        return losses, indices
    if reduction_type not in INDEX_COUNTS:
        raise ValueError(f"Unknown reduction_type: {reduction_type}")
    index_tensors = (indices,) if isinstance(indices, Tensor) else indices
    if not isinstance(index_tensors, (tuple, list)) or not all(
        isinstance(x, Tensor) for x in index_tensors
    ):
        raise TypeError(f"{reduction_type} indices must be tensors")
    if len(index_tensors) != INDEX_COUNTS[reduction_type]:
        count = INDEX_COUNTS[reduction_type]
        raise ValueError(f"{reduction_type} indices must contain {count} tensors")
    if losses.ndim != 1 or any(x.ndim != 1 for x in index_tensors):
        raise ValueError("losses and indices must be one-dimensional")
    if any(len(x) != len(losses) for x in index_tensors):
        raise ValueError("losses and indices must have equal lengths")
    return losses, indices
