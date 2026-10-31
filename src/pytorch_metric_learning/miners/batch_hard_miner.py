from torch import Tensor

from ..distances import BaseDistance
from ..types import IndicesTuple, TripletIndices
from .batch_easy_hard_miner import BatchEasyHardMiner


class BatchHardMiner(BatchEasyHardMiner):
    def __init__(
        self,
        *,
        distance: BaseDistance | None = None,
        tuples_per_anchor: int | None = None,
    ) -> None:
        super().__init__(
            pos_strategy="hard",
            neg_strategy="hard",
            distance=distance,
            tuples_per_anchor=tuples_per_anchor,
        )

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
        *,
        mat: Tensor | None = None,
    ) -> IndicesTuple:
        a1, p, _, n = super().mine(
            embeddings, labels, indices_tuple, ref_emb, ref_labels, mat=mat
        )
        return TripletIndices(a1, p, n)
