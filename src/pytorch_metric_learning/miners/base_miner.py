import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple
from ..utils import _common


class BaseMiner(torch.nn.Module):
    def __init__(self, *, distance: BaseDistance | None = None) -> None:
        super().__init__()
        self.distance = distance if distance is not None else self.default_distance()

    def default_distance(self) -> BaseDistance:
        return LpDistance()

    def mine(
        self, embeddings: Tensor, labels: Tensor, ref_emb: Tensor, ref_labels: Tensor
    ) -> IndicesTuple:
        raise NotImplementedError

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
    ) -> IndicesTuple:
        with torch.no_grad():
            embeddings, labels, ref_emb, ref_labels = prepare_mining_inputs(
                embeddings, labels, ref_emb, ref_labels
            )
            output = self.mine(embeddings, labels, ref_emb, ref_labels)
        validate_output(output)
        return output


def prepare_mining_inputs(
    embeddings: Tensor,
    labels: Tensor,
    ref_emb: Tensor | None,
    ref_labels: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    prepared = _common.prepare_inputs(embeddings, labels, ref_emb, ref_labels)
    embeddings, prepared_labels, ref_emb, prepared_ref_labels = prepared
    if prepared_labels is None or prepared_ref_labels is None:
        raise ValueError("Mining requires labels and ref_labels")
    return embeddings, prepared_labels, ref_emb, prepared_ref_labels


def validate_output(output: IndicesTuple) -> None:
    if not isinstance(output, (tuple, list)) or not all(
        isinstance(index, Tensor) for index in output
    ):
        raise TypeError("Mining must return a tuple of index tensors")
    if any(index.ndim != 1 for index in output):
        raise ValueError("Mining indices must be one-dimensional tensors")
    if len(output) == 3:
        if not len(output[0]) == len(output[1]) == len(output[2]):
            raise ValueError("Triplet mining must return three tensors of equal length")
    elif len(output) == 4:
        if len(output[0]) != len(output[1]) or len(output[2]) != len(output[3]):
            raise ValueError(
                "Pair mining must return anchor and partner indices of equal length"
            )
    else:
        raise ValueError("Mining must return three triplet or four pair index tensors")
