from torch import Tensor

from ..distances import BaseDistance
from ..types import IndicesTuple, PairIndices
from ..utils import _common, indexing
from .base_miner import BaseMiner


class PairMarginMiner(BaseMiner):
    supports_mat = True

    def __init__(
        self,
        pos_margin: float = 0.2,
        neg_margin: float = 0.8,
        *,
        distance: BaseDistance | None = None,
    ) -> None:
        super().__init__(distance=distance)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

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
        pos_mask = self.distance.margin(mat[a1, p], self.pos_margin) > 0
        neg_mask = self.distance.margin(self.neg_margin, mat[a2, n]) > 0
        return PairIndices(a1[pos_mask], p[pos_mask], a2[neg_mask], n[neg_mask])
