import math
from typing import Unpack

import torch
from torch import Tensor

from ..types import ProxyLossOptions
from .large_margin_softmax_loss import _CosineMarginSoftmaxLoss


class ArcFaceLoss(_CosineMarginSoftmaxLoss):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float = 0.5,
        scale: float = 64,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        super().__init__(num_classes, embedding_size, margin, scale, **options)

    def target_margins(self, cosine: Tensor, labels: Tensor) -> Tensor:
        return cosine.new_tensor(self.margin)

    def modify_cosine_of_target_classes(
        self, cosine_of_target_classes: Tensor, labels: Tensor
    ) -> Tensor:
        margins = self.target_margins(cosine_of_target_classes, labels)
        angles = self.get_angles(cosine_of_target_classes)
        return torch.where(
            angles <= math.pi - margins,
            torch.cos(angles + margins),
            torch.cos(angles) - margins * torch.sin(margins),
        )
