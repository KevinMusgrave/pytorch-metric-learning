#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from ..distances import DotProductSimilarity

class NPairsLoss(BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/abs/1708.01682
    Args:
        l2_reg_weight: The L2 regularizer weight (multiplier)
    """
    def __init__(self, l2_reg_weight=0, **kwargs):
        super().__init__(**kwargs)
        self.l2_reg_weight = l2_reg_weight
        self.add_to_recordable_attributes(name="num_pairs", is_stat=True)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, embeddings, labels, indices_tuple):
        self.avg_embedding_norm = torch.mean(torch.norm(embeddings, p=2, dim=1))
        anchor_idx, positive_idx = lmu.convert_to_pos_pairs_with_unique_labels(indices_tuple, labels)
        self.num_pairs = len(anchor_idx)
        if self.num_pairs == 0:
            return self.zero_losses()
        anchors, positives = embeddings[anchor_idx], embeddings[positive_idx]
        targets = torch.arange(self.num_pairs).to(embeddings.device)
        sim_mat = self.distance(anchors, positives)
        if not self.distance.is_inverted:
            sim_mat = -sim_mat
        loss_dict = {"loss": {"losses": self.cross_entropy(sim_mat, targets), "indices": anchor_idx, "reduction_type": "element"}}
        if self.l2_reg_weight > 0:
            l2_reg = torch.norm(embeddings, p=2, dim=1)
            loss_dict["l2_reg"] = {"losses": l2_reg * self.l2_reg_weight, "indices": c_f.torch_arange_from_size(embeddings), "reduction_type": "element"}
        return loss_dict

    def sub_loss_names(self):
        return ["loss", "l2_reg"]

    def get_default_distance(self):
        return DotProductSimilarity()