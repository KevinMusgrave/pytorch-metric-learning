import math

import torch
from torch import Tensor

from ..distances import BaseDistance
from ..types import IndicesTuple, PairIndices
from ..utils import indexing
from .base_miner import BaseMiner, prepare_mining_inputs, validate_output


class HDCMiner(BaseMiner):
    def __init__(
        self, filter_percentage: float = 0.5, *, distance: BaseDistance | None = None
    ) -> None:
        super().__init__(distance=distance)
        if not 0 < filter_percentage <= 1:
            raise ValueError("filter_percentage must be in (0, 1]")
        self.filter_percentage = filter_percentage

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
        candidates: IndicesTuple | None = None,
    ) -> IndicesTuple:
        with torch.no_grad():
            embeddings, labels, ref_emb, ref_labels = prepare_mining_inputs(
                embeddings, labels, ref_emb, ref_labels
            )
            pairs = indexing.convert_to_pairs(candidates, labels, ref_labels)
            output = self.hardest_pairs(embeddings, ref_emb, pairs)
        validate_output(output)
        return output

    def mine(
        self, embeddings: Tensor, labels: Tensor, ref_emb: Tensor, ref_labels: Tensor
    ) -> IndicesTuple:
        pairs = indexing.get_all_pairs_indices(labels, ref_labels)
        return self.hardest_pairs(embeddings, ref_emb, pairs)

    def hardest_pairs(
        self, embeddings: Tensor, ref_emb: Tensor, pairs: PairIndices
    ) -> PairIndices:
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = pairs
        a1, p = self.hardest(mat, a1, p, largest=not self.distance.is_inverted)
        a2, n = self.hardest(mat, a2, n, largest=self.distance.is_inverted)
        return a1, p, a2, n

    def hardest(
        self, mat: Tensor, anchors: Tensor, others: Tensor, largest: bool
    ) -> tuple[Tensor, Tensor]:
        if len(anchors) == 0:
            return anchors, others
        k = math.ceil(self.filter_percentage * len(anchors))
        _, idx = torch.topk(mat[anchors, others], k=k, largest=largest)
        return anchors[idx], others[idx]
