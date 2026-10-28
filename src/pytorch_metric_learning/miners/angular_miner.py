import torch
from torch import Tensor

from ..types import IndicesTuple
import math


from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseMiner


class AngularMiner(BaseMiner):
    def __init__(self, angle=20, **kwargs):
        super().__init__(**kwargs)
        c_f.check_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.angle = math.radians(angle)

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
        ref_emb: Tensor,
        ref_labels: Tensor,
    ) -> IndicesTuple:
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels, ref_labels
        )
        anchors, positives, negatives = (
            embeddings[anchor_idx],
            ref_emb[positive_idx],
            ref_emb[negative_idx],
        )
        centers = (anchors + positives) / 2
        ap_dist = self.distance.pairwise_distance(anchors, positives)
        nc_dist = self.distance.pairwise_distance(negatives, centers)
        angles = torch.atan(ap_dist / (2 * nc_dist))
        threshold_condition = angles > self.angle
        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )
