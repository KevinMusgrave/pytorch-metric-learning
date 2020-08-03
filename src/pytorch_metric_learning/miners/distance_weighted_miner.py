#! /usr/bin/env python3

from .base_miner import BaseTupleMiner
import torch
from ..utils import loss_and_miner_utils as lmu


# adapted from
# https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/model.py
class DistanceWeightedMiner(BaseTupleMiner):
    def __init__(self, cutoff, nonzero_loss_cutoff, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = float(cutoff)
        self.nonzero_loss_cutoff = float(nonzero_loss_cutoff)
        self.add_to_recordable_attributes(list_of_names=["cutoff", "nonzero_loss_cutoff"], is_stat=False)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        dtype, device = embeddings.dtype, embeddings.device
        d = float(embeddings.size(1))
        mat = self.distance(embeddings, ref_emb)
        
        # Cut off to avoid high variance.
        mat = torch.clamp(mat, min=self.cutoff)

        # See the first equation from Section 4 of the paper
        log_weights = (2.0 - d) * torch.log(mat) - ((d - 3) / 2) * torch.log(1.0 - 0.25 * (mat ** 2.0))

        inf_or_nan = torch.isinf(log_weights) | torch.isnan(log_weights)
        # Subtract max(log(distance)) for stability.
        weights = torch.exp(log_weights - torch.max(log_weights[~inf_or_nan]))

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = torch.ones_like(weights)
        same_class = labels.unsqueeze(1) == ref_labels.unsqueeze(0)
        mask[same_class] = 0

        weights = weights * mask * ((mat < self.nonzero_loss_cutoff).type(dtype))
        weights[inf_or_nan] = 0
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        np_weights = weights.cpu().numpy()
        return lmu.get_random_triplet_indices(labels, ref_labels=ref_labels, weights=np_weights)
