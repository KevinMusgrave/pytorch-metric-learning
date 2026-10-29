import torch
from torch import Tensor

from ..types import LossDict, LossTerm
from ..utils._common import zero_loss


class BaseReducer(torch.nn.Module):
    def forward(
        self, loss_dict: LossDict, embeddings: Tensor, labels: Tensor | None
    ) -> Tensor:
        reduced = (self.reduce(term, embeddings, labels) for term in loss_dict.values())
        return sum(reduced, start=zero_loss(embeddings))

    def reduce(
        self, term: LossTerm, embeddings: Tensor, labels: Tensor | None
    ) -> Tensor:
        losses, anchors = validate_term(term)
        if "divisor" in term:
            raise ValueError(
                f"{type(self).__name__} cannot apply a loss term's divisor; "
                "reduce this loss with DivisorReducer"
            )
        if anchors is None:
            return losses
        if losses.numel() == 0:
            return zero_loss(embeddings)
        return self.element_reduction(losses, anchors, embeddings, labels)

    def element_reduction(
        self,
        losses: Tensor,
        anchors: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        raise NotImplementedError


def validate_term(term: LossTerm) -> tuple[Tensor, Tensor | None]:
    losses, indices = term["losses"], term["indices"]
    if not isinstance(losses, Tensor):
        raise TypeError("losses must be a tensor")
    if indices is None:
        if losses.numel() != 1:
            raise ValueError("A loss term without indices must contain one value")
        return losses.reshape(()), None
    index_tensors = (indices,) if isinstance(indices, Tensor) else indices
    if (
        not isinstance(index_tensors, tuple)
        or len(index_tensors) not in (1, 2, 3)
        or not all(isinstance(x, Tensor) for x in index_tensors)
    ):
        raise TypeError("indices must be a tensor or a tuple of two or three tensors")
    if losses.ndim != 1 or any(x.ndim != 1 for x in index_tensors):
        raise ValueError("losses and indices must be one-dimensional")
    if any(len(x) != len(losses) for x in index_tensors):
        raise ValueError("losses and indices must have equal lengths")
    return losses, index_tensors[0]
