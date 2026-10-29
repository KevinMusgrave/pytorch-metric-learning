from typing import Unpack

from ..distances import BaseDistance, SNRDistance
from ..types import LossOptions
from ..utils import _common
from .contrastive_loss import ContrastiveLoss


class SignalToNoiseRatioContrastiveLoss(ContrastiveLoss):
    def __init__(
        self,
        pos_margin: float = 0.0,
        neg_margin: float = 1.0,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(pos_margin, neg_margin, **options)
        _common.check_distance_type(self, SNRDistance)

    def default_distance(self) -> BaseDistance:
        return SNRDistance()
