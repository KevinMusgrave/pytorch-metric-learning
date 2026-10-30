from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..reducers import BaseReducer, DivisorReducer
from ..types import LossDict, RegularizerOptions
from ..utils import _common
from .base_regularizer import BaseRegularizer


class SparseCentersRegularizer(BaseRegularizer):
    same_class_mask: Tensor

    def __init__(
        self,
        num_classes: int,
        centers_per_class: int,
        **options: Unpack[RegularizerOptions],
    ) -> None:
        super().__init__(**options)
        if centers_per_class <= 1:
            raise ValueError("centers_per_class must be greater than one")
        _common.check_distance_type(self, CosineSimilarity)
        self.set_class_masks(num_classes, centers_per_class)

    def compute_loss(self, x: Tensor) -> LossDict:
        center_similarities = self.distance(x)
        small_val = _common.small_val(x.dtype)
        center_similarities_masked = torch.clamp(
            2.0 * center_similarities[self.same_class_mask], max=2
        )
        divisor = 2 * torch.sum(self.same_class_mask)
        reg = torch.sqrt(2.0 + small_val - center_similarities_masked)
        return {
            "loss": {
                "losses": reg,
                "indices": _common.arange_like(reg),
                "divisor": divisor,
            }
        }

    def set_class_masks(self, num_classes: int, centers_per_class: int) -> None:
        total_num_centers = num_classes * centers_per_class
        same_class_mask = torch.zeros(
            total_num_centers, total_num_centers, dtype=torch.bool
        )
        for i in range(num_classes):
            s, e = i * centers_per_class, (i + 1) * centers_per_class
            curr_block = torch.ones(centers_per_class, centers_per_class)
            curr_block = torch.triu(curr_block, diagonal=1)
            same_class_mask[s:e, s:e] = curr_block
        self.register_buffer("same_class_mask", same_class_mask, persistent=False)

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

    def default_reducer(self) -> BaseReducer:
        return DivisorReducer()
