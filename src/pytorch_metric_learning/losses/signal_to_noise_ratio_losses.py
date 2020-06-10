import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from .base_metric_loss_function import BaseMetricLossFunction
from ..reducers import AvgNonZeroReducer


def SNR_dist(x, y, dim):
    return torch.var(x-y, dim=dim) / torch.var(x, dim=dim)


class SignalToNoiseRatioContrastiveLoss(BaseMetricLossFunction):

    def __init__(self, pos_margin, neg_margin, regularizer_weight, **kwargs):
        super().__init__(**kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.regularizer_weight = regularizer_weight
        self.add_to_recordable_attributes(name="feature_distance_from_zero_mean_distribution", is_stat=True)
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        a1, p, a2, n = lmu.convert_to_pairs(indices_tuple, labels)
        pos_loss, neg_loss, reg_loss = 0, 0, 0
        if len(a1) > 0:
            pos_loss = self.get_per_pair_loss(embeddings[a1], embeddings[p], self.pos_margin, 1)
        if len(a2) > 0:
            neg_loss = self.get_per_pair_loss(embeddings[a2], embeddings[n], self.neg_margin, -1)
        self.feature_distance_from_zero_mean_distribution = torch.mean(torch.abs(torch.sum(embeddings, dim=1)))
        if self.regularizer_weight > 0:
            reg_loss = self.regularizer_weight * self.feature_distance_from_zero_mean_distribution
        
        loss_dict = {"pos_loss": {"losses": pos_loss, "indices": (a1, p), "reduction_type": "pos_pair"}, 
                    "neg_loss": {"losses": neg_loss, "indices": (a2, n), "reduction_type": "neg_pair"},
                    "reg_loss": {"losses": reg_loss, "indices": c_f.torch_arange_from_size(embeddings), "reduction_type": "already_reduced"}}

        return loss_dict


    def get_per_pair_loss(self, anchors, others, margin, before_relu_multiplier):
        d = SNR_dist(anchors, others, dim=1)
        return torch.nn.functional.relu((d-margin)*before_relu_multiplier)

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def sub_loss_names(self):
        return ["pos_loss", "neg_loss", "reg_loss"]