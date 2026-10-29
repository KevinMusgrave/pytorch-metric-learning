from collections.abc import Callable
from typing import Unpack

import torch
from torch import Tensor

from ..regularizers import BaseRegularizer
from ..types import LossDict, LossOptions
from .base_metric_loss_function import BaseMetricLossFunction, regularization_term


class BaseProxyLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_weights: int,
        embedding_size: int,
        *,
        weight_regularizer: BaseRegularizer | None = None,
        weight_reg_weight: float = 1.0,
        weight_init_func: Callable[[Tensor], object] | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.weight = torch.nn.Parameter(torch.empty(num_weights, embedding_size))
        (weight_init_func or self.default_weight_init)(self.weight)

    def default_weight_init(self, weight: Tensor) -> None:
        torch.nn.init.normal_(weight)

    def regularization_terms(self, embeddings: Tensor) -> LossDict:
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.weight
            )
        return terms
