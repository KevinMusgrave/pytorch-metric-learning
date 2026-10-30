import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, PairIndices
from ..utils import _common, indexing
from .base_miner import BaseMiner


class MultiSimilarityMiner(BaseMiner):
    def __init__(
        self, epsilon: float = 0.1, *, distance: BaseDistance | None = None
    ) -> None:
        super().__init__(distance=distance)
        self.epsilon = epsilon

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> IndicesTuple:
        mat = self.distance(embeddings, ref_emb)
        if mat.numel() == 0:
            return _common.empty_pairs(mat.device)
        pairs = indexing.convert_to_pairs(indices_tuple, labels, ref_labels)
        pos_mask, neg_mask = indexing._pair_masks(
            pairs, (mat.shape[0], mat.shape[1]), mat.device
        )
        high, low = _common.pos_inf(mat.dtype), _common.neg_inf(mat.dtype)
        if self.distance.is_inverted:
            pos_sorted, pos_sorted_idx = torch.sort(mat.masked_fill(~pos_mask, high))
            neg_sorted, neg_sorted_idx = torch.sort(mat.masked_fill(~neg_mask, low))
            hard_pos = torch.where(
                pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1)
            )
            hard_neg = torch.where(
                neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1)
            )
        else:
            pos_sorted, pos_sorted_idx = torch.sort(mat.masked_fill(~pos_mask, low))
            neg_sorted, neg_sorted_idx = torch.sort(mat.masked_fill(~neg_mask, high))
            hard_pos = torch.where(
                pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1)
            )
            hard_neg = torch.where(
                neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1)
            )
        a1, a2 = hard_pos[0], hard_neg[0]
        p = pos_sorted_idx[a1, hard_pos[1]]
        n = neg_sorted_idx[a2, hard_neg[1]]
        return PairIndices(a1, p, a2, n)

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
