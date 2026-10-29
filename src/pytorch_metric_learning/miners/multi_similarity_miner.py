import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple
from ..utils import _common, indexing
from .base_miner import BaseMiner


class MultiSimilarityMiner(BaseMiner):
    def __init__(
        self, epsilon: float = 0.1, *, distance: BaseDistance | None = None
    ) -> None:
        super().__init__(distance=distance)
        self.epsilon = epsilon

    def mine(
        self, embeddings: Tensor, labels: Tensor, ref_emb: Tensor, ref_labels: Tensor
    ) -> IndicesTuple:
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = indexing.get_all_pairs_indices(labels, ref_labels)
        if len(a1) == 0 or len(a2) == 0:
            empty = torch.empty(0, device=labels.device, dtype=torch.long)
            return empty, empty.clone(), empty.clone(), empty.clone()

        dtype = mat.dtype
        pos_ignore = (
            _common.pos_inf(dtype)
            if self.distance.is_inverted
            else _common.neg_inf(dtype)
        )
        neg_ignore = (
            _common.neg_inf(dtype)
            if self.distance.is_inverted
            else _common.pos_inf(dtype)
        )
        mat_pos_sorting = mat.clone()
        mat_neg_sorting = mat.clone()
        mat_pos_sorting[a2, n] = pos_ignore
        mat_neg_sorting[a1, p] = neg_ignore
        if embeddings is ref_emb:
            mat_pos_sorting.fill_diagonal_(pos_ignore)
            mat_neg_sorting.fill_diagonal_(neg_ignore)

        pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
        neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)

        if self.distance.is_inverted:
            hard_pos_idx = torch.where(
                pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1)
            )
        else:
            hard_pos_idx = torch.where(
                pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1)
            )

        a1 = hard_pos_idx[0]
        p = pos_sorted_idx[a1, hard_pos_idx[1]]
        a2 = hard_neg_idx[0]
        n = neg_sorted_idx[a2, hard_neg_idx[1]]
        return a1, p, a2, n

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
