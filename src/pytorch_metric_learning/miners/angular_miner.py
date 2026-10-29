import math

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple
from ..utils import _common, indexing
from .base_miner import BaseMiner


class AngularMiner(BaseMiner):
    def __init__(
        self, angle: float = 20, *, distance: BaseDistance | None = None
    ) -> None:
        super().__init__(distance=distance)
        _common.check_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.angle = math.radians(angle)

    def mine(
        self, embeddings: Tensor, labels: Tensor, ref_emb: Tensor, ref_labels: Tensor
    ) -> IndicesTuple:
        anchor_idx, positive_idx, negative_idx = indexing.get_all_triplets_indices(
            labels, ref_labels
        )
        anchors = embeddings[anchor_idx]
        positives = ref_emb[positive_idx]
        negatives = ref_emb[negative_idx]
        centers = (anchors + positives) / 2
        ap_dist = self.distance.pairwise_distance(anchors, positives)
        nc_dist = self.distance.pairwise_distance(negatives, centers)
        angles = torch.atan(ap_dist / (2 * nc_dist))
        keep = angles > self.angle
        return anchor_idx[keep], positive_idx[keep], negative_idx[keep]
