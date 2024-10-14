import numpy as np
import torch

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseMiner


class AngularMiner(BaseMiner):
    """
    Returns triplets that form an angle greater than some threshold (angle).
    The angle is computed as defined in the angular loss paper:
    https://arxiv.org/abs/1708.01682
    """

    def __init__(self, angle=20, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.angle = np.radians(angle)
        self.add_to_recordable_attributes(list_of_names=["angle"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=[
                "average_angle",
                "average_angle_above_threshold",
                "average_angle_below_threshold",
                "min_angle",
                "max_angle",
                "std_of_angle",
            ],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
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
        self.set_stats(angles, threshold_condition)
        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )

    def set_stats(self, angles, threshold_condition):
        if self.collect_stats:
            with torch.no_grad():
                if len(angles) > 0:
                    self.average_angle = np.degrees(torch.mean(angles).item())
                    self.min_angle = np.degrees(torch.min(angles).item())
                    self.max_angle = np.degrees(torch.max(angles).item())
                    self.std_of_angle = np.degrees(torch.std(angles).item())
                if torch.sum(threshold_condition) > 0:
                    self.average_angle_above_threshold = np.degrees(
                        torch.mean(angles[threshold_condition]).item()
                    )
                negated_condition = ~threshold_condition
                if torch.sum(negated_condition) > 0:
                    self.average_angle_below_threshold = np.degrees(
                        torch.mean(angles[~threshold_condition]).item()
                    )
