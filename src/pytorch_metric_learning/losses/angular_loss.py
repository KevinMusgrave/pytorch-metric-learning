#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import numpy as np
import torch
from ..utils import loss_and_miner_utils as lmu

class AngularLoss(BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/abs/1708.01682
    Args:
        alpha: The angle (as described in the paper), specified in degrees.
    """
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.tensor(np.radians(alpha))
        self.add_to_recordable_attributes(list_of_names=["average_angle"], is_stat=True, optional=True)
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        anchors, positives, keep_mask, anchor_idx = self.set_stats_get_pairs(embeddings, labels, indices_tuple)
        if anchors is None: 
            return self.zero_losses()

        sq_tan_alpha = torch.tan(self.alpha) ** 2
        ap_dot = torch.sum(anchors * positives, dim=1, keepdim=True)
        ap_matmul_embeddings = torch.matmul((anchors + positives),(embeddings.unsqueeze(2)))
        ap_matmul_embeddings = ap_matmul_embeddings.squeeze(2).t()

        final_form = (4 * sq_tan_alpha * ap_matmul_embeddings) - (2 * (1 + sq_tan_alpha) * ap_dot)
        final_form = self.maybe_modify_loss(final_form)
        losses = lmu.logsumexp(final_form, keep_mask=keep_mask, add_one=True)
        return {"loss": {"losses": losses, "indices": anchor_idx, "reduction_type": "element"}}

    def set_stats_get_pairs(self, embeddings, labels, indices_tuple):
        a1, p, a2, _ = lmu.convert_to_pairs(indices_tuple, labels)
        if len(a1) == 0 or len(a2) == 0:
            return [None]*4
        anchors, positives = embeddings[a1], embeddings[p]
        keep_mask = (labels[a1].unsqueeze(1) != labels.unsqueeze(0)).float()

        centers = (anchors + positives) / 2
        ap_dist = torch.nn.functional.pairwise_distance(anchors, positives, 2)
        nc_dist = torch.norm(centers - embeddings.unsqueeze(1), p=2, dim=2).t()
        angles = torch.atan(ap_dist.unsqueeze(1) / (2*nc_dist))
        average_angle = torch.sum(angles*keep_mask) / torch.sum(keep_mask)
        self.average_angle = np.degrees(average_angle.item())
        return anchors, positives, keep_mask, a1

    def maybe_modify_loss(self, x):
        return x
