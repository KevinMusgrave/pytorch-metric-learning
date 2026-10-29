import math
from collections.abc import Callable
from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction, regularization_term


class CosineMarginSoftmaxLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float,
        scale: float,
        *,
        weight_regularizer: BaseRegularizer | None = None,
        weight_reg_weight: float = 1.0,
        weight_init_func: Callable[[Tensor], object] | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        _common.check_distance_type(self, CosineSimilarity)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.weight = torch.nn.Parameter(torch.empty(num_classes, embedding_size))
        (weight_init_func or torch.nn.init.normal_)(self.weight)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

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
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        _common.check_parameter_placement(self, self.weight, embeddings)
        miner_weights = indexing.convert_to_weights(
            indices_tuple, labels, dtype=embeddings.dtype
        )
        mask = torch.nn.functional.one_hot(labels, self.num_classes).bool()
        cosine = self.get_cosine(embeddings)
        cosine_of_target_classes = cosine[mask]
        modified = self.modify_cosine_of_target_classes(cosine_of_target_classes)
        diff = (modified - cosine_of_target_classes).unsqueeze(1)
        logits = self.scale_logits(cosine + mask * diff, embeddings)
        loss = self.cross_entropy(logits, labels) * miner_weights
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
        return CosineSimilarity()

    def get_logits(self, embeddings: Tensor) -> Tensor:
        return self.scale_logits(self.get_cosine(embeddings), embeddings)


class LargeMarginSoftmaxLoss(CosineMarginSoftmaxLoss):
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
        *,
        weight_regularizer: BaseRegularizer | None = None,
        weight_reg_weight: float = 1.0,
        weight_init_func: Callable[[Tensor], object] | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        if not isinstance(margin, int) or margin < 1:
            raise ValueError("margin must be a positive integer")
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
