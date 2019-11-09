#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import numpy as np
import torch
from ..utils import loss_and_miner_utils as lmu


class AngularNPairsLoss(BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/abs/1708.01682
    Args:
        alpha: The angle (as described in the paper), specified in degrees.
    """
    def __init__(self, alpha, **kwargs):
        self.alpha = torch.tensor(np.radians(alpha))
        self.maybe_modify_loss = lambda x: x
        self.num_anchors = 0
        self.avg_embedding_norm = 0
        self.record_these = ["num_anchors", "avg_embedding_norm"]
        super().__init__(**kwargs)

    def compute_loss(self, embeddings, labels, indices_tuple):
        self.avg_embedding_norm = torch.mean(torch.norm(embeddings, p=2, dim=1))
        anchor_idx, positive_idx = lmu.convert_to_pos_pairs_with_unique_labels(indices_tuple, labels)
        self.num_anchors = len(anchor_idx)
        if self.num_anchors == 0:
            return 0

        anchors, positives = embeddings[anchor_idx], embeddings[positive_idx]
        points_per_anchor = (self.num_anchors - 1) * 2

        alpha = self.maybe_mask_param(self.alpha, labels[anchor_idx])
        sq_tan_alpha = torch.tan(alpha) ** 2

        xa_xp = torch.sum(anchors * positives, dim=1, keepdim=True)
        term2_multiplier = 2 * (1 + sq_tan_alpha)
        term2 = term2_multiplier * xa_xp

        a_p_summed = anchors + positives
        inside_exp = []
        mask = torch.ones(self.num_anchors).to(embeddings.device) - torch.eye(self.num_anchors).to(embeddings.device)
        term1_multiplier = 4 * sq_tan_alpha

        for negatives in [anchors, positives]:
            term1 = term1_multiplier * torch.matmul(a_p_summed, torch.t(negatives))
            inside_exp.append(term1 - term2.repeat(1, self.num_anchors))
            inside_exp[-1] = inside_exp[-1] * mask

        inside_exp_final = torch.zeros((self.num_anchors, points_per_anchor + 1)).to(embeddings.device)

        for i in range(self.num_anchors):
            indices = np.concatenate((np.arange(0, i), np.arange(i + 1, inside_exp[0].size(1))))
            inside_exp_final[i, : points_per_anchor // 2] = inside_exp[0][i, indices]
        inside_exp_final[:, points_per_anchor // 2 :] = inside_exp[1]
        inside_exp_final = self.maybe_modify_loss(inside_exp_final)

        return torch.mean(torch.logsumexp(inside_exp_final, dim=1))

    def create_learnable_parameter(self, init_value):
        return super().create_learnable_parameter(init_value, unsqueeze=True)


class AngularLoss(AngularNPairsLoss):
    def compute_loss(self, embeddings, labels, indices_tuple):
        self.avg_embedding_norm = torch.mean(torch.norm(embeddings, p=2, dim=1))
        anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(indices_tuple, labels)
        self.num_anchors = len(anchor_idx)
        if self.num_anchors == 0:
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        alpha = self.maybe_mask_param(self.alpha, labels[anchor_idx])
        sq_tan_alpha = torch.tan(alpha) ** 2
        term1 = 4 * sq_tan_alpha * torch.sum((anchors + positives) * negatives, dim=1, keepdim=True)
        term2 = 2 * (1 + sq_tan_alpha) * torch.sum(anchors * positives, dim=1, keepdim=True)
        final_form = torch.cat([term1 - term2, torch.zeros(term1.size(0), 1).to(embeddings.device)], dim=1)
        final_form = self.maybe_modify_loss(final_form)
        return torch.mean(torch.logsumexp(final_form, dim=1))
