from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..reducers import BaseReducer, DivisorReducer
from ..types import IndicesTuple, LossDict, ProxyLossOptions
from ..utils import _common, indexing
from .base_proxy_loss import BaseProxyLoss


class ProxyAnchorLoss(BaseProxyLoss):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float = 0.1,
        alpha: float = 32,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        super().__init__(num_classes, embedding_size, **options)
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha

    def default_weight_init(self, weight: Tensor) -> None:
        torch.nn.init.kaiming_normal_(weight, mode="fan_out")

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.ref_not_supported(ref_emb)
        _common.check_parameter_placement(self, self.weight, embeddings)
        cos = self.get_logits(embeddings)
        weights = indexing.convert_to_weights(indices_tuple, labels, dtype=cos.dtype)
        log_weights = torch.log(weights).unsqueeze(1)
        pos_mask = torch.nn.functional.one_hot(labels, self.num_classes).bool()
        weighted = weights.unsqueeze(1) > 0
        pos_exp = self.distance.margin(cos, self.margin)
        neg_exp = self.distance.margin(-self.margin, cos)
        pos_term = _common.logsumexp(
            self.alpha * pos_exp + log_weights,
            keep_mask=pos_mask & weighted,
            add_one=True,
            dim=0,
        )
        neg_term = _common.logsumexp(
            self.alpha * neg_exp + log_weights,
            keep_mask=~pos_mask & weighted,
            add_one=True,
            dim=0,
        )
        loss_indices = _common.arange_like(self.weight)
        return {
            "pos_loss": {
                "losses": pos_term.squeeze(0),
                "indices": loss_indices,
                "divisor": pos_mask.any(dim=0).sum(),
            },
            "neg_loss": {
                "losses": neg_term.squeeze(0),
                "indices": loss_indices,
                "divisor": self.num_classes,
            },
        }

    def default_reducer(self) -> BaseReducer:
        return DivisorReducer()

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

    def get_logits(self, embeddings: Tensor) -> Tensor:
        return self.distance(embeddings, self.weight)
