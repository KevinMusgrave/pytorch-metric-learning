#! /usr/bin/env python3

from .base_miner import BasePostGradientMiner
from ..utils import loss_and_miner_utils as lmu
import torch


class MultiSimilarityMiner(BasePostGradientMiner):
    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def mine(self, embeddings, labels):
        self.n = embeddings.size(0)
        self.index_list = torch.arange(self.n).to(embeddings.device)
        self.sim_mat = lmu.sim_mat(embeddings)
        return self.compute_indices(labels)

    def compute_indices(self, labels):
        empty_tensor = torch.tensor([]).long().to(labels.device)
        a1_idx, p_idx, a2_idx, n_idx = [empty_tensor], [empty_tensor], [empty_tensor], [empty_tensor]
        for i in range(self.n):
            pos_indices = (
                ((labels == labels[i]) & (self.index_list != i)).nonzero().flatten()
            )
            neg_indices = (labels != labels[i]).nonzero().flatten()

            if pos_indices.size(0) == 0 or neg_indices.size(0) == 0:
                continue

            pos_sorted, pos_sorted_idx = torch.sort(self.sim_mat[i][pos_indices])
            neg_sorted, neg_sorted_idx = torch.sort(self.sim_mat[i][neg_indices])
            neg_sorted_filtered_idx = (
                (neg_sorted + self.epsilon > pos_sorted[0]).nonzero().flatten()
            )
            pos_sorted_filtered_idx = (
                (pos_sorted - self.epsilon < neg_sorted[-1]).nonzero().flatten()
            )

            pos_indices = pos_indices[pos_sorted_idx][pos_sorted_filtered_idx]
            neg_indices = neg_indices[neg_sorted_idx][neg_sorted_filtered_idx]

            if len(pos_indices) > 0:
                a1_idx.append(torch.ones_like(pos_indices) * i)
                p_idx.append(pos_indices)
            if len(neg_indices) > 0:
                a2_idx.append(torch.ones_like(neg_indices) * i)
                n_idx.append(neg_indices)

        return [torch.cat(idx) for idx in [a1_idx, p_idx, a2_idx, n_idx]]
