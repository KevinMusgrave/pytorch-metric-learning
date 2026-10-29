import math
from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, LossDict, ProxyLossOptions
from ..utils import _common, indexing
from .base_proxy_loss import BaseProxyLoss


class _CosineMarginSoftmaxLoss(BaseProxyLoss):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float,
        scale: float,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        super().__init__(num_classes, embedding_size, **options)
        _common.check_distance_type(self, CosineSimilarity)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

    def get_cosine(self, embeddings: Tensor) -> Tensor:
        return self.distance(embeddings, self.weight)

    @staticmethod
    def get_angles(cosine_of_target_classes: Tensor) -> Tensor:
        return torch.acos(torch.clamp(cosine_of_target_classes, -1, 1))

    def modify_cosine_of_target_classes(
        self, cosine_of_target_classes: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def scale_logits(self, logits: Tensor, embeddings: Tensor) -> Tensor:
        return logits * self.scale

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
        cosine = self.get_cosine(embeddings)
        targets = labels.unsqueeze(1)
        modified = self.modify_cosine_of_target_classes(
            cosine.gather(1, targets).squeeze(1)
        )
        logits = self.scale_logits(
            cosine.scatter(1, targets, modified.unsqueeze(1).to(cosine.dtype)),
            embeddings,
        )
        weights = indexing.convert_to_weights(indices_tuple, labels, dtype=logits.dtype)
        loss = F.cross_entropy(logits, labels, reduction="none") * weights
        return {"loss": {"losses": loss, "indices": _common.arange_like(embeddings)}}

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

    def get_logits(self, embeddings: Tensor) -> Tensor:
        return self.scale_logits(self.get_cosine(embeddings), embeddings)


class LargeMarginSoftmaxLoss(_CosineMarginSoftmaxLoss):
    n_range: Tensor
    margin_choose_n: Tensor
    cos_powers: Tensor
    alternating: Tensor

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: int = 4,
        scale: float = 1,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        if not isinstance(margin, int) or margin < 1:
            raise ValueError("margin must be a positive integer")
        super().__init__(num_classes, embedding_size, margin, scale, **options)
        n_range = torch.arange(margin // 2 + 1)
        coefficients = [float(math.comb(margin, 2 * n)) for n in n_range]
        self.register_buffer("n_range", n_range, persistent=False)
        self.register_buffer(
            "margin_choose_n", torch.tensor(coefficients), persistent=False
        )
        self.register_buffer("cos_powers", margin - 2 * n_range, persistent=False)
        self.register_buffer("alternating", (-1) ** n_range, persistent=False)

    def get_cos_with_margin(self, cosine: Tensor) -> Tensor:
        cosine = cosine.unsqueeze(1)
        cos_powered = cosine**self.cos_powers
        sin_powered = (1 - cosine**2) ** self.n_range
        terms = self.alternating * self.margin_choose_n * cos_powered * sin_powered
        return torch.sum(terms, dim=1)

    def modify_cosine_of_target_classes(
        self, cosine_of_target_classes: Tensor
    ) -> Tensor:
        cos_with_margin = self.get_cos_with_margin(cosine_of_target_classes)
        angles = self.get_angles(cosine_of_target_classes)
        with torch.no_grad():
            k = (angles / (math.pi / self.margin)).floor()
        return ((-1) ** k) * cos_with_margin - (2 * k)

    def scale_logits(self, logits: Tensor, embeddings: Tensor) -> Tensor:
        embedding_norms = self.distance.get_norm(embeddings)
        weight_norms = self.distance.get_norm(self.weight)
        product_of_magnitudes = weight_norms.unsqueeze(0) * embedding_norms.unsqueeze(1)
        return logits * product_of_magnitudes * self.scale
