import torch
from torch import Tensor

from ..types import IndicesTuple, LossDict, PairIndices
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class GenericPairLoss(BaseMetricLossFunction):
    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        _common.labels_or_indices_tuple_required(labels, indices_tuple)
        pairs = indexing.convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) == 0 for x in pairs):
            return self.zero_losses(embeddings)
        return self.pair_loss(self.distance(embeddings, ref_emb), pairs)

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        raise NotImplementedError

    @staticmethod
    def pair_values(mat: Tensor, pairs: PairIndices) -> tuple[Tensor, Tensor]:
        a1, p, a2, n = pairs
        return mat[a1, p], mat[a2, n]

    @staticmethod
    def pair_masks(mat: Tensor, pairs: PairIndices) -> tuple[Tensor, Tensor]:
        a1, p, a2, n = pairs
        pos_mask = torch.zeros_like(mat, dtype=torch.bool)
        neg_mask = torch.zeros_like(mat, dtype=torch.bool)
        pos_mask[a1, p] = True
        neg_mask[a2, n] = True
        return pos_mask, neg_mask
