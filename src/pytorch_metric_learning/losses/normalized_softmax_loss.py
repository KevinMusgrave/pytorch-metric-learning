from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor

from ..distances import BaseDistance, DotProductSimilarity
from ..types import IndicesTuple, LossDict, ProxyLossOptions
from ..utils import _common, indexing
from .base_proxy_loss import BaseProxyLoss


class NormalizedSoftmaxLoss(BaseProxyLoss):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        temperature: float = 0.05,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        super().__init__(num_classes, embedding_size, **options)
        self.temperature = temperature

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
        logits, dtype = self._get_logits(embeddings)
        if torch.is_autocast_enabled(embeddings.device.type):
            dtype = logits.dtype
        weights = indexing.convert_to_weights(indices_tuple, labels, dtype=logits.dtype)
        loss = F.cross_entropy(logits, labels, reduction="none") * weights
        return {
            "loss": {
                "losses": loss.to(dtype),
                "indices": _common.arange_like(embeddings),
            }
        }

    def default_distance(self) -> BaseDistance:
        return DotProductSimilarity(normalize_embeddings=True)

    def get_logits(self, embeddings: Tensor) -> Tensor:
        return self._get_logits(embeddings)[0]

    def _get_logits(self, embeddings: Tensor) -> tuple[Tensor, torch.dtype]:
        normalized_weight = self.distance.normalize(self.weight, dim=1)
        logits = self.distance(embeddings, normalized_weight)
        dtype = logits.dtype
        logits = logits.to(_common.accumulation_dtype(logits.dtype)) / self.temperature
        if not self.distance.is_inverted:
            logits = -logits
        return logits, dtype
