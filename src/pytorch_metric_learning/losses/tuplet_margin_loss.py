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
        a1, p, a2, _ = pairs
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        dtype = _common.accumulation_dtype(mat.dtype)
        bound = 1 - 4 * torch.finfo(pos_pairs.dtype).eps
        pos_angles = torch.acos(pos_pairs.to(dtype).clamp(-bound, bound))
        pos_with_margin = torch.cos(pos_angles - math.radians(self.margin))
        neg_lse = _common.segment_logsumexp(
            self.scale * neg_pairs.to(dtype), a2, mat.shape[0]
        )
        losses = torch.nn.functional.softplus(
            neg_lse[a1] - self.scale * pos_with_margin
        )
        return {"loss": {"losses": losses.to(mat.dtype), "indices": (a1, p)}}

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
