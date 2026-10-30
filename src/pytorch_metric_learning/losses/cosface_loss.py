from typing import Unpack

from torch import Tensor

from ..types import ProxyLossOptions
from .large_margin_softmax_loss import _CosineMarginSoftmaxLoss


class CosFaceLoss(_CosineMarginSoftmaxLoss):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float = 0.35,
        scale: float = 64,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        super().__init__(num_classes, embedding_size, margin, scale, **options)

    def modify_cosine_of_target_classes(
        self, cosine_of_target_classes: Tensor, labels: Tensor
    ) -> Tensor:
        return cosine_of_target_classes - self.margin
