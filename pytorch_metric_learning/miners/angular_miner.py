#! /usr/bin/env python3

from .base_miner import BasePostGradientMiner
import torch
from ..utils import loss_and_miner_utils as lmu
import numpy as np

class AngularMiner(BasePostGradientMiner):
    """
    Returns triplets that form an angle greater than some threshold (angle).
    The angle is computed as defined in the angular loss paper:
    https://arxiv.org/abs/1708.01682
    """
    def __init__(self, angle, **kwargs):
        super().__init__(**kwargs)
        self.angle = torch.tensor(np.radians(angle))
        self.average_angle = 0
        self.average_angle_above_threshold = 0
        self.average_angle_below_threshold = 0
        self.min_angle, self.max_angle = 0, 0
        self.record_these += ["average_angle", 
                            "average_angle_above_threshold", 
                            "average_angle_below_threshold",
                            "min_angle", "max_angle", "std_of_angle"]

    def mine(self, embeddings, labels):
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(labels)
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        centers = (anchors + positives) / 2
        ap_dist = torch.nn.functional.pairwise_distance(anchors, positives, 2)
        nc_dist = torch.nn.functional.pairwise_distance(negatives, centers, 2)
        angles = torch.atan(ap_dist / (2*nc_dist))
        threshold_condition = angles > self.angle.to(angles.device)
        self.set_stats(angles, threshold_condition)
        return anchor_idx[threshold_condition], positive_idx[threshold_condition], negative_idx[threshold_condition]

    def set_stats(self, angles, threshold_condition):
        self.average_angle = np.degrees(torch.mean(angles).item())
        self.min_angle = np.degrees(torch.min(angles).item())
        self.max_angle = np.degrees(torch.max(angles).item())
        self.std_of_angle = np.degrees(torch.std(angles).item())
        self.average_angle_above_threshold = 0
        self.average_angle_below_threshold = 0
        if torch.sum(threshold_condition) > 0:
            self.average_angle_above_threshold = np.degrees(torch.mean(angles[threshold_condition]).item())
        negated_condition = ~threshold_condition
        if torch.sum(negated_condition) > 0:
            self.average_angle_below_threshold = np.degrees(torch.mean(angles[~threshold_condition]).item())