import torch
import torch.nn.functional as F

from ..distances import CosineSimilarity
from ..reducers import AvgNonZeroReducer
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction

def l2_norm(v):
    fnorm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-6
    v = v.div(fnorm.expand_as(v))
    return v

class InstanceLoss(BaseMetricLossFunction):
    """
    Implementation of  Dual-Path Convolutional Image-Text Embeddings with Instance Loss, ACM TOMM 2020 
    https://arxiv.org/abs/1711.05535 
    using cross-entropy loss for every sample if label is not available, else use given label.
    """
    def __init__(self, gamma=64, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, CosineSimilarity)
        self.gamma = gamma
        self.add_to_recordable_attributes(
            list_of_names=["gamma"],
            is_stat=False,
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        normed_feature = l2_norm(embeddings)
        sim1 = torch.mm(normed_feature*self.gamma, torch.t(normed_feature)) 
        if labels is None:
            sim_label = torch.arange(sim1.size(0)).cuda().detach()
        else:
            _, sim_label = torch.unique(labels, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label, reduction="none")
        return {
            "loss": {
                "losses": loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return CosineSimilarity()
