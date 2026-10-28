import torch
from torch import Tensor

from ..types import IndicesTuple, LossDict
import math


from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class AngularLoss(BaseMetricLossFunction):
    def __init__(self, alpha=40, **kwargs):
        super().__init__(**kwargs)
        c_f.check_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.alpha = torch.tensor(math.radians(alpha), dtype=torch.float64)

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        c_f.labels_required(labels)
        anchors, positives, keep_mask, anchor_idx, positive_idx = self.get_pairs(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        if anchors is None:
            return self.zero_losses()

        sq_tan_alpha = torch.tan(self.alpha) ** 2
        ap_dot = torch.sum(anchors * positives, dim=1, keepdim=True)
        negatives = self.distance.normalize(ref_emb)
        ap_matmul_embeddings = torch.matmul(
            (anchors + positives), (negatives.unsqueeze(2))
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
        return anchors, positives, keep_mask, a1, p
