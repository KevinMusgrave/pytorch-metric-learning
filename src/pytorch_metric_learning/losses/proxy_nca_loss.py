from collections.abc import Callable
from typing import Unpack

import torch
from torch import Tensor

from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common
from .base_metric_loss_function import regularization_term
from .nca_loss import NCALoss


class ProxyNCALoss(NCALoss):
    proxy_labels: Tensor

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        softmax_scale: float = 1,
        *,
        weight_regularizer: BaseRegularizer | None = None,
        weight_reg_weight: float = 1.0,
        weight_init_func: Callable[[Tensor], object] | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(softmax_scale, **options)
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.weight = torch.nn.Parameter(torch.empty(num_classes, embedding_size))
        (weight_init_func or torch.nn.init.normal_)(self.weight)
        self.register_buffer(
            "proxy_labels", torch.arange(num_classes), persistent=False
        )

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        _common.check_parameter_placement(self, self.weight, embeddings)
        embeddings = embeddings.to(dtype=self.weight.dtype)
        return self.nca_computation(
            embeddings, self.weight, labels, self.proxy_labels, indices_tuple
        )

    def regularization_terms(self, embeddings: Tensor) -> LossDict:
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.weight
            )
        return terms

    def get_logits(self, embeddings: Tensor) -> Tensor:
        _common.check_parameter_placement(self, self.weight, embeddings)
        embeddings = embeddings.to(dtype=self.weight.dtype)
        return self.distance(embeddings, self.weight)
