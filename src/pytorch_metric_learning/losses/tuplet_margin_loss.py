import math
from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import LossDict, LossOptions, PairIndices
from ..utils import _common
from .generic_pair_loss import GenericPairLoss


class TupletMarginLoss(GenericPairLoss):
    def __init__(
        self, margin: float = 5.73, scale: float = 64, **options: Unpack[LossOptions]
    ) -> None:
        super().__init__(**options)
        _common.check_distance_type(self, CosineSimilarity)
        self.margin = margin
        self.scale = scale

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        a1, p, a2, _ = pairs
        if len(a1) == 0 or len(a2) == 0:
            return self.zero_losses(mat)
        bound = 1 - 4 * torch.finfo(pos_pairs.dtype).eps
        pos_angles = torch.acos(pos_pairs.clamp(-bound, bound))
        pos_pairs = torch.cos(pos_angles - math.radians(self.margin)).unsqueeze(1)
        inside_exp = self.scale * (neg_pairs.unsqueeze(0) - pos_pairs)
        keep_mask = a2.unsqueeze(0) == a1.unsqueeze(1)
        loss = _common.logsumexp(inside_exp, keep_mask=keep_mask, add_one=True, dim=1)
        return {
            "loss": {
                "losses": loss.squeeze(1),
                "indices": (a1, p),
                "reduction_type": "pos_pair",
            }
        }

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
