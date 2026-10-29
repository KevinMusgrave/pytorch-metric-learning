from collections.abc import Callable
from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..reducers import BaseReducer, DivisorReducer
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction, regularization_term


def default_proxy_init(proxies: Tensor) -> None:
    torch.nn.init.kaiming_normal_(proxies, mode="fan_out")


class ProxyAnchorLoss(BaseMetricLossFunction):
    sub_loss_names = ("pos_loss", "neg_loss")

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float = 0.1,
        alpha: float = 32,
        *,
        weight_regularizer: BaseRegularizer | None = None,
        weight_reg_weight: float = 1.0,
        weight_init_func: Callable[[Tensor], object] | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.weight = torch.nn.Parameter(torch.empty(num_classes, embedding_size))
        (weight_init_func or default_proxy_init)(self.weight)
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha

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
        dtype = embeddings.dtype
        miner_weights = indexing.convert_to_weights(indices_tuple, labels, dtype=dtype)
        miner_weights = miner_weights.unsqueeze(1) - 1
        cos = self.get_logits(embeddings)
        pos_mask = torch.nn.functional.one_hot(labels, self.num_classes).bool()
        neg_mask = ~pos_mask
        pos_exp = self.distance.margin(cos, self.margin)
        neg_exp = self.distance.margin(-self.margin, cos)
        pos_term = _common.logsumexp(
            (self.alpha * pos_exp) + miner_weights,
            keep_mask=pos_mask,
            add_one=True,
            dim=0,
        )
        neg_term = _common.logsumexp(
            (self.alpha * neg_exp) + miner_weights,
            keep_mask=neg_mask,
            add_one=True,
            dim=0,
        )
        loss_indices = _common.arange_like(self.weight)
        return {
            "pos_loss": {
                "losses": pos_term.squeeze(0),
                "indices": loss_indices,
                "reduction_type": "element",
                "divisor": pos_mask.any(dim=0).sum(),
            },
            "neg_loss": {
                "losses": neg_term.squeeze(0),
                "indices": loss_indices,
                "reduction_type": "element",
                "divisor": self.num_classes,
            },
        }

    def regularization_terms(self, embeddings: Tensor) -> LossDict:
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.weight
            )
        return terms

    def default_reducer(self) -> BaseReducer:
        return DivisorReducer()

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

    def get_logits(self, embeddings: Tensor) -> Tensor:
        return self.distance(embeddings, self.weight)
