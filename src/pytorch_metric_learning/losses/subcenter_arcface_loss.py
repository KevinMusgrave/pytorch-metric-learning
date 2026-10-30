import math
from typing import Unpack

import torch
from torch import Tensor

from ..types import ProxyLossOptions, SubCenterOutliers
from ..utils import _common
from .arcface_loss import ArcFaceLoss


class SubCenterArcFaceLoss(ArcFaceLoss):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float = 0.5,
        scale: float = 64,
        sub_centers: int = 3,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        super().__init__(
            num_classes * sub_centers, embedding_size, margin, scale, **options
        )
        self.sub_centers = sub_centers
        self.num_classes = num_classes

    def get_cosine(self, embeddings: Tensor) -> Tensor:
        cosine = self.distance(embeddings, self.weight)
        cosine = cosine.view(-1, self.num_classes, self.sub_centers)
        return cosine.max(dim=2).values

    def get_outliers(
        self,
        embeddings: Tensor,
        labels: Tensor,
        threshold_radians: float = math.radians(75),
    ) -> SubCenterOutliers:
        _common.check_shapes(embeddings, labels)
        _common.check_parameter_placement(self, self.weight, embeddings)
        cos_threshold = math.cos(threshold_radians)
        with torch.no_grad():
            centers = self.weight.view(self.num_classes, self.sub_centers, -1)
            similarities = self.distance(embeddings, self.weight)
            similarities = similarities.view(-1, self.num_classes, self.sub_centers)
            nearest = similarities[torch.arange(len(labels)), labels].argmax(dim=1)
            votes = torch.zeros(
                self.num_classes,
                self.sub_centers,
                dtype=torch.long,
                device=labels.device,
            )
            votes.index_put_(
                (labels, nearest), torch.ones_like(nearest), accumulate=True
            )
            dominant = votes.argmax(dim=1)
            dominant_centers = centers[torch.arange(self.num_classes), dominant]
            dominant_similarity = similarities[
                torch.arange(len(labels)), labels, dominant[labels]
            ]
            outliers = torch.where(dominant_similarity < cos_threshold)[0]
        return SubCenterOutliers(outliers, dominant_centers)
