from typing import Unpack

import torch
from torch import Tensor

from ..types import IndicesTuple, LossDict, LossOptions, PairIndices
from ..utils import _common
from .generic_pair_loss import GenericPairLoss


class LiftedStructureLoss(GenericPairLoss):
    def __init__(
        self,
        neg_margin: float = 1.0,
        pos_margin: float = 0.0,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        _common.ref_not_supported(ref_emb)
        return super().compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        a1, p, a2, _ = pairs
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        dtype = _common.accumulation_dtype(mat.dtype)
        remaining_pos_margin = self.distance.margin(pos_pairs, self.pos_margin)
        remaining_neg_margin = self.distance.margin(self.neg_margin, neg_pairs)
        neg_lse = _common.segment_logsumexp(
            remaining_neg_margin.to(dtype), a2, mat.shape[0]
        )
        neg_pairs_loss = torch.logaddexp(neg_lse[a1], neg_lse[p])
        loss_per_pos_pair = torch.relu(neg_pairs_loss + remaining_pos_margin) ** 2 / 2
        return {"loss": {"losses": loss_per_pos_pair.to(mat.dtype), "indices": (a1, p)}}


class GeneralizedLiftedStructureLoss(GenericPairLoss):
    def __init__(
        self,
        neg_margin: float = 1.0,
        pos_margin: float = 0.0,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        pos_mask, neg_mask = self.pair_masks(mat, pairs)
        remaining_pos_margin = self.distance.margin(mat, self.pos_margin)
        remaining_neg_margin = self.distance.margin(self.neg_margin, mat)
        pos_loss = _common.logsumexp(
            remaining_pos_margin, keep_mask=pos_mask, add_one=False
        )
        neg_loss = _common.logsumexp(
            remaining_neg_margin, keep_mask=neg_mask, add_one=False
        )
        return {
            "loss": {
                "losses": torch.relu(pos_loss + neg_loss).squeeze(1),
                "indices": _common.arange_like(mat),
            }
        }
