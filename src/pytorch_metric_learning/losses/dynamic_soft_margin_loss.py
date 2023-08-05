import torch
import numpy as np

from ..reducers import MeanReducer

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction

def find_hard_negatives(dmat):
    """
    a = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ...
    """


    pos = dmat.diag()
    dmat.fill_diagonal_(np.inf)

    min_a, _ = torch.min(dmat, dim=0)
    min_p, _ = torch.min(dmat, dim=1)
    neg = torch.min(min_a, min_p)
    return pos, neg


class DynamicSoftMarginLoss(BaseMetricLossFunction):
    """Loss function with dynamical margin parameter introduced in https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_Local_Descriptors_With_a_CDF-Based_Dynamic_Soft_Margin_ICCV_2019_paper.pdf

    Args:
        min_val: minimum significative value for `d_pos - d_neg`
        num_bins: number of equally spaced bins for the partition of the interval [min_val, :math:`+\infty`]
        momentum: weight assigned to the histogram computed from the current batch
    """
    def __init__(self, min_val=-2.0, num_bins=10, momentum=0.01, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, LpDistance, normalize_embeddings=True, p=2)
        self.min_val = min_val
        self.num_bins = int(num_bins)
        self.delta = 2*abs(min_val) / num_bins
        self.momentum = momentum
        self.hist_ = torch.zeros((num_bins,))
        self.add_to_recordable_attributes(list_of_names=["num_bins"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        self.hist_ = c_f.to_device(self.hist_, tensor=embeddings, dtype=embeddings.dtype)

        if labels is None:
            loss = self.compute_loss_without_labels(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        else:
            loss = self.compute_loss_with_labels(embeddings, labels, indices_tuple, ref_emb, ref_labels)

        if len(loss) == 0:
            return self.zero_losses()

        self.update_histogram(loss)
        loss = self.weigh_loss(loss)
        loss = loss.mean()
        return {
                "loss": {
                    "losses": loss,
                    "indices": None,
                    "reduction_type": "already_reduced",
                }
            }
    
    def compute_loss_without_labels(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        r, c = mat.size()

        min_a, min_p = torch.zeros(max(r, c)), torch.zeros(max(r, c))   # Check for unequal number of anchors and positives
        min_a = c_f.to_device(min_a, tensor=embeddings, dtype=embeddings.dtype)
        min_p = c_f.to_device(min_p, tensor=embeddings, dtype=embeddings.dtype)
        min_a[:c], _ = torch.min(mat, dim=0)
        min_p[:r], _ = torch.min(mat, dim=1)

        d_pos = torch.zeros(max(r, c))
        d_pos = c_f.to_device(d_pos, tensor=embeddings, dtype=embeddings.dtype)
        d_pos[:min(r, c)], d_neg = mat.diag(), torch.min(min_a, min_p)
        return d_pos - d_neg

    def compute_loss_with_labels(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(indices_tuple, labels, ref_labels, t_per_anchor="all")     # Use all instead of t_per_anchor=1 to be deterministic
        mat = self.distance(embeddings, ref_emb)
        d_pos, d_neg = mat[anchor_idx, positive_idx], mat[anchor_idx, negative_idx]
        return d_pos - d_neg

    def update_histogram(self, data):
        idx, alpha = torch.floor((data - self.min_val) / self.delta).to(dtype=torch.long), torch.frac((data - self.min_val) / self.delta)
        momentum = self.momentum if self.hist_.sum() != 0 else 1.0
        self.hist_ = torch.scatter_add((1.0 - momentum) * self.hist_, 0, idx, momentum*(1 - alpha))
        self.hist_ = torch.scatter_add(self.hist_, 0, idx+1, momentum*alpha)
        self.hist_ /= self.hist_.sum()

    def weigh_loss(self, data):
        CDF = torch.cumsum(self.hist_, 0)
        idx = torch.floor((data - self.min_val) / self.delta).to(dtype=torch.long)
        return CDF[idx]*data
