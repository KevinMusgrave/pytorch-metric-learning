import math

import torch
from torch import Tensor

from ..distances import BaseDistance
from ..types import IndicesTuple, PairIndices
from ..utils import _common, indexing
from .base_miner import BaseMiner


class HDCMiner(BaseMiner):
    supports_mat = True

    def __init__(
        self, filter_percentage: float = 0.5, *, distance: BaseDistance | None = None
    ) -> None:
        super().__init__(distance=distance)
        if not 0 < filter_percentage <= 1:
            raise ValueError("filter_percentage must be in (0, 1]")
        self.filter_percentage = filter_percentage

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
        mat = _common.resolve_matrix(self.distance, embeddings, ref_emb, mat)
        a1, p, a2, n = indexing.convert_to_pairs(indices_tuple, labels, ref_labels)
        a1, p = self.hardest(mat, a1, p, largest=not self.distance.is_inverted)
        a2, n = self.hardest(mat, a2, n, largest=self.distance.is_inverted)
        return PairIndices(a1, p, a2, n)

    def hardest(
        self, mat: Tensor, anchors: Tensor, others: Tensor, largest: bool
    ) -> tuple[Tensor, Tensor]:
        if len(anchors) == 0:
            return anchors, others
        k = math.ceil(self.filter_percentage * len(anchors))
        _, idx = torch.topk(mat[anchors, others], k=k, largest=largest)
        return anchors[idx], others[idx]
