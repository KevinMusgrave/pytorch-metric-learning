import torch
from torch import Tensor

from ..distances import BaseDistance
from ..types import IndicesTuple, PairIndices
from ..utils import _common, indexing
from .base_miner import BaseMiner


class UniformHistogramMiner(BaseMiner):
    supports_mat = True

    def __init__(
        self,
        num_bins: int = 100,
        pos_per_bin: int = 10,
        neg_per_bin: int = 10,
        *,
        distance: BaseDistance | None = None,
        tuples_per_anchor: int | None = None,
    ) -> None:
        super().__init__(distance=distance, tuples_per_anchor=tuples_per_anchor)
        self.num_bins = num_bins
        self.pos_per_bin = pos_per_bin
        self.neg_per_bin = neg_per_bin

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
        if len(a1) > 0:
            a1, p = self.uniformly_distributed_pairs(
                mat[a1, p], a1, p, self.pos_per_bin
            )
        if len(a2) > 0:
            a2, n = self.uniformly_distributed_pairs(
                mat[a2, n], a2, n, self.neg_per_bin
            )
        return PairIndices(a1, p, a2, n)

    def uniformly_distributed_pairs(
        self, distances: Tensor, anchors: Tensor, others: Tensor, num_pairs: int
    ) -> tuple[Tensor, Tensor]:
        low, high = distances.min(), distances.max()
        width = (high - low).clamp_min(_common.small_val(distances.dtype))
        bins = ((distances - low) / width * self.num_bins).long()
        bins = bins.clamp_(max=self.num_bins - 1)
        occupied = torch.unique(bins)
        membership = (bins.unsqueeze(0) == occupied.unsqueeze(1)).to(torch.float32)
        chosen = torch.multinomial(membership, num_pairs, replacement=True).flatten()
        return anchors[chosen], others[chosen]
