from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple, LossDict, ProxyLossOptions
from ..utils import _common, indexing
from .base_proxy_loss import BaseProxyLoss
from .nca_loss import _nca_terms


class ProxyNCALoss(BaseProxyLoss):
    proxy_labels: Tensor

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        softmax_scale: float = 1,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        super().__init__(num_classes, embedding_size, **options)
        self.softmax_scale = softmax_scale
        self.register_buffer(
            "proxy_labels", torch.arange(num_classes), persistent=False
        )

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
        weights = indexing.convert_to_weights(
            indices_tuple, labels, dtype=_common.accumulation_dtype(embeddings.dtype)
        )
        return _nca_terms(
            self.get_logits(embeddings),
            labels,
            self.proxy_labels,
            weights,
            self.softmax_scale,
            self.distance.is_inverted,
            exclude_self=False,
        )

    def default_distance(self) -> BaseDistance:
        return LpDistance(power=2)

    def get_logits(self, embeddings: Tensor) -> Tensor:
        _common.check_parameter_placement(self, self.weight, embeddings)
        embeddings = embeddings.to(dtype=self.weight.dtype)
        return self.distance(embeddings, self.weight)
