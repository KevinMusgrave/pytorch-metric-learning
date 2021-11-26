import numpy as np
import torch

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class AngularLoss(BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/abs/1708.01682
    Args:
        alpha: The angle (as described in the paper), specified in degrees.
    """

    def __init__(self, alpha=40, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.alpha = torch.tensor(np.radians(alpha))
        self.add_to_recordable_attributes(list_of_names=["alpha"], is_stat=False)
        self.add_to_recordable_attributes(list_of_names=["average_angle"], is_stat=True)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        anchors, positives, keep_mask, anchor_idx, positive_idx = self.get_pairs(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        if anchors is None:
            return self.zero_losses()

        sq_tan_alpha = torch.tan(self.alpha) ** 2
        ap_dot = torch.sum(anchors * positives, dim=1, keepdim=True)
        ap_matmul_embeddings = torch.matmul(
            (anchors + positives), (ref_emb.unsqueeze(2))
        )
        ap_matmul_embeddings = ap_matmul_embeddings.squeeze(2).t()

        final_form = (4 * sq_tan_alpha * ap_matmul_embeddings) - (
            2 * (1 + sq_tan_alpha) * ap_dot
        )
        losses = lmu.logsumexp(final_form, keep_mask=keep_mask, add_one=True)
        return {
            "loss": {
                "losses": losses,
                "indices": (anchor_idx, positive_idx),
                "reduction_type": "pos_pair",
            }
        }

    def get_pairs(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        a1, p, a2, _ = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
        if len(a1) == 0 or len(a2) == 0:
            return [None] * 5
        anchors = self.distance.normalize(embeddings[a1])
        positives = self.distance.normalize(ref_emb[p])
        keep_mask = labels[a1].unsqueeze(1) != ref_labels.unsqueeze(0)
        self.set_stats(anchors, positives, embeddings, ref_emb, keep_mask)
        return anchors, positives, keep_mask, a1, p

    def set_stats(self, anchors, positives, embeddings, ref_emb, keep_mask):
        if self.collect_stats:
            with torch.no_grad():
                centers = (anchors + positives) / 2
                ap_dist = self.distance.pairwise_distance(anchors, positives)
                nc_dist = self.distance.get_norm(
                    centers - ref_emb.unsqueeze(1), dim=2
                ).t()
                angles = torch.atan(ap_dist.unsqueeze(1) / (2 * nc_dist))
                average_angle = torch.sum(angles[keep_mask]) / torch.sum(keep_mask)
                self.average_angle = np.degrees(average_angle.item())
