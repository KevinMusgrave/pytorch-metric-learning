#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu


class NPairsLoss(BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/abs/1708.01682
    Args:
        l2_reg_weight: The L2 regularizer weight (multiplier)
    """
    def __init__(self, l2_reg_weight=0, **kwargs):
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.l2_reg_weight = l2_reg_weight
        self.num_pairs = 0
        self.avg_embedding_norm = 0
        self.record_these = ["num_pairs", "avg_embedding_norm"]
        super().__init__(**kwargs)

    def compute_loss(self, embeddings, labels, indices_tuple):
        self.avg_embedding_norm = torch.mean(torch.norm(embeddings, p=2, dim=1))
        anchor_idx, positive_idx = lmu.convert_to_pos_pairs_with_unique_labels(indices_tuple, labels)
        self.num_pairs = len(anchor_idx)
        if self.num_pairs == 0:
            return 0
        anchors, positives = embeddings[anchor_idx], embeddings[positive_idx]
        targets = torch.arange(self.num_pairs).to(embeddings.device)
        sim_mat = torch.matmul(anchors, positives.t())
        s_loss = self.cross_entropy(sim_mat, targets)
        if self.l2_reg_weight > 0:
            l2_reg = torch.mean(torch.norm(embeddings, p=2, dim=1))
            return s_loss + l2_reg * self.l2_reg_weight
        return s_loss
