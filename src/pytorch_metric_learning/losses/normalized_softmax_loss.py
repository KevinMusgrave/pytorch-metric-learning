from collections.abc import Callable
from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, DotProductSimilarity
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction, regularization_term


class NormalizedSoftmaxLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        temperature: float = 0.05,
        *,
        weight_regularizer: BaseRegularizer | None = None,
        weight_reg_weight: float = 1.0,
        weight_init_func: Callable[[Tensor], object] | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.temperature = temperature
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.weight = torch.nn.Parameter(torch.empty(num_classes, embedding_size))
        (weight_init_func or torch.nn.init.normal_)(self.weight)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

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
        miner_weights = indexing.convert_to_weights(
            indices_tuple, labels, dtype=embeddings.dtype
        )
        loss = self.cross_entropy(self.get_logits(embeddings), labels) * miner_weights
        return {
            "loss": {
                "losses": loss,
                "indices": _common.arange_like(embeddings),
                "reduction_type": "element",
            }
        }

    def regularization_terms(self, embeddings: Tensor) -> LossDict:
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.weight
            )
        return terms

    def default_distance(self) -> BaseDistance:
        return DotProductSimilarity()

    def get_logits(self, embeddings: Tensor) -> Tensor:
        normalized_weight = self.distance.normalize(self.weight, dim=1)
        logits = self.distance(embeddings, normalized_weight) / self.temperature
        if not self.distance.is_inverted:
            logits = -logits
        return logits
