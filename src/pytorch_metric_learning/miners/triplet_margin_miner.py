from torch import Tensor

from ..distances import BaseDistance
from ..types import IndicesTuple, MiningStrategy, TripletIndices
from ..utils import _common, indexing
from .base_miner import BaseMiner


class TripletMarginMiner(BaseMiner):
    supports_mat = True

    def __init__(
        self,
        margin: float = 0.2,
        type_of_triplets: MiningStrategy = "all",
        *,
        distance: BaseDistance | None = None,
    ) -> None:
        super().__init__(distance=distance)
        if type_of_triplets not in ("all", "hard", "semihard", "easy"):
            raise ValueError(
                'type_of_triplets must be "all", "hard", "semihard" or "easy"'
            )
        self.margin = margin
        self.type_of_triplets = type_of_triplets

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
        anchor_idx, positive_idx, negative_idx = indexing.convert_to_triplets(
            indices_tuple, labels, ref_labels
        )
        mat = _common.resolve_matrix(self.distance, embeddings, ref_emb, mat)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = self.distance.margin(an_dist, ap_dist)
        if self.type_of_triplets == "easy":
            keep = triplet_margin > self.margin
        else:
            keep = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                keep &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                keep &= triplet_margin > 0
        return TripletIndices(anchor_idx[keep], positive_idx[keep], negative_idx[keep])
