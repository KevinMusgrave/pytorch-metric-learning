import torch.nn.functional as F

from ..utils.loss_and_miner_utils import convert_to_pairs
from .base_metric_loss_function import BaseMetricLossFunction
from ..distances import CosineSimilarity
from ..utils import common_functions as c_f

class ThresholdConsistentMarginLoss(BaseMetricLossFunction):
    """
        Implements the TCM loss from: https://arxiv.org/abs/2307.04047
    """
    
    def __init__(
        self, base_loss, lambda_plus=1.0, lambda_minus=1.0, margin_plus=0.9, margin_minus=0.5, **kwargs
    ):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, CosineSimilarity)
        self.base_loss = base_loss
        self.lambda_plus = lambda_plus
        self.lambda_minus = lambda_minus
        self.margin_plus = margin_plus
        self.margin_minus = margin_minus


    def get_default_distance(self):
        return CosineSimilarity()

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        ap, p, an, n = convert_to_pairs(indices_tuple, labels, ref_labels)

        # calculate the similarities for positive and negative pairs
        ap, p = embeddings[ap], embeddings[p]
        an, n = embeddings[an], embeddings[n]

        pos_sims = F.cosine_similarity(ap, p)
        neg_sims = F.cosine_similarity(an, n)

        # calculate the positive part
        s_lte_m = (pos_sims <= self.margin_plus)
        tcm_pos_num = ((self.margin_plus - pos_sims) * s_lte_m).sum()
        tcm_pos_denom = s_lte_m.sum()
        pos_tcm = 0 if s_lte_m.sum() == 0 else tcm_pos_num / tcm_pos_denom

        # calculate the negative part
        s_gte_m = (neg_sims >= self.margin_minus)
        tcm_neg_num = ((neg_sims - self.margin_minus) * s_gte_m).sum()
        tcm_neg_denom = s_gte_m.sum() 
        neg_tcm = 0 if s_gte_m.sum() == 0 else tcm_neg_num / tcm_neg_denom

        # add the components for final loss
        tcm_loss = self.lambda_plus * pos_tcm + self.lambda_minus * neg_tcm
        base_loss = self.base_loss(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        total_loss = base_loss + tcm_loss
        return {
            "loss": {
                "losses": total_loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }