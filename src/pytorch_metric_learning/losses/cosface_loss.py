from collections.abc import Callable
from typing import Unpack

from torch import Tensor

from ..regularizers import BaseRegularizer
from ..types import LossOptions
from .large_margin_softmax_loss import CosineMarginSoftmaxLoss


class CosFaceLoss(CosineMarginSoftmaxLoss):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float = 0.35,
        scale: float = 64,
        *,
        weight_regularizer: BaseRegularizer | None = None,
        weight_reg_weight: float = 1.0,
        weight_init_func: Callable[[Tensor], object] | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(
            num_classes,
            embedding_size,
            margin,
            scale,
            weight_regularizer=weight_regularizer,
            weight_reg_weight=weight_reg_weight,
            weight_init_func=weight_init_func,
            **options,
        )

    def modify_cosine_of_target_classes(
        self, cosine_of_target_classes: Tensor
    ) -> Tensor:
        return cosine_of_target_classes - self.margin
