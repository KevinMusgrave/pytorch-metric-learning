#! /usr/bin/env python3

from .base_miner import BasePostGradientMiner
import torch
from ..utils import loss_and_miner_utils as lmu


# adapted from
# https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/model.py
class DistanceWeightedMiner(BasePostGradientMiner):
    def __init__(self, cutoff, nonzero_loss_cutoff, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = float(cutoff)
        self.nonzero_loss_cutoff = float(nonzero_loss_cutoff)

    def mine(self, embeddings, labels):
        label_set = torch.unique(labels)
        n, d = embeddings.size()

        dist_mat = lmu.dist_mat(embeddings)
        dist_mat = dist_mat + torch.eye(dist_mat.size(0)).to(embeddings.device)  
        # so that we don't get log(0). We mask the diagonal out later anyway
        # Cut off to avoid high variance.
        dist_mat = torch.max(dist_mat, torch.tensor(self.cutoff).to(dist_mat.device))

        # Subtract max(log(distance)) for stability.
        # See the first equation from Section 4 of the paper
        log_weights = (2.0 - float(d)) * torch.log(dist_mat) - (
            float(d - 3) / 2
        ) * torch.log(1.0 - 0.25 * (dist_mat ** 2.0))
        weights = torch.exp(log_weights - torch.max(log_weights))

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = torch.ones(weights.size()).to(embeddings.device)
        for i in label_set:
            idx = (labels == i).nonzero().squeeze(1)
            mask[torch.meshgrid(idx, idx)] = 0

        weights = weights * mask * ((dist_mat < self.nonzero_loss_cutoff).float())
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        np_weights = weights.cpu().numpy()
        return lmu.get_random_triplet_indices(labels, weights=np_weights)
