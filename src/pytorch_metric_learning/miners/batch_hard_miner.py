from torch import Tensor

from ..distances import BaseDistance
from ..types import IndicesTuple
from .batch_easy_hard_miner import BatchEasyHardMiner


class BatchHardMiner(BatchEasyHardMiner):
    def __init__(self, *, distance: BaseDistance | None = None) -> None:
        super().__init__(pos_strategy="hard", neg_strategy="hard", distance=distance)

    def mine(
        self, embeddings: Tensor, labels: Tensor, ref_emb: Tensor, ref_labels: Tensor
    ) -> IndicesTuple:
        a1, p, _, n = super().mine(embeddings, labels, ref_emb, ref_labels)
        return a1, p, n
