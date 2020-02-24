#! /usr/bin/env python3

from .base_miner import BasePostGradientMiner
import torch
from ..utils import loss_and_miner_utils as lmu


class BatchHardMiner(BasePostGradientMiner):
    def __init__(self, use_similarity=False, squared_distances=False, **kwargs):
        super().__init__(**kwargs)
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.add_to_recordable_attributes(list_of_names=["hardest_triplet_dist", "hardest_pos_pair_dist", "hardest_neg_pair_dist"])

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = lmu.get_pairwise_mat(embeddings, ref_emb, self.use_similarity, self.squared_distances)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, ref_labels)

        pos_func = self.get_min_per_row if self.use_similarity else self.get_max_per_row
        neg_func = self.get_max_per_row if self.use_similarity else self.get_min_per_row

        hardest_positive_dist, hardest_positive_indices = pos_func(mat, a1_idx, p_idx)
        hardest_negative_dist, hardest_negative_indices = neg_func(mat, a2_idx, n_idx) 
        self.set_stats(hardest_positive_dist, hardest_negative_dist)
        
        return torch.arange(mat.size(0)).to(hardest_positive_indices.device), hardest_positive_indices, hardest_negative_indices

    def get_max_per_row(self, mat, anchor_idx, other_idx):
        mask = torch.zeros_like(mat)
        mask[anchor_idx, other_idx] = 1
        mat_masked = mat * mask 
        return torch.max(mat_masked, dim=1)

    def get_min_per_row(self, mat, anchor_idx, other_idx):
        mask = torch.ones_like(mat) * float('inf')
        mask[anchor_idx, other_idx] = 1
        mat_masked = mat * mask
        mat_masked[torch.isnan(mat_masked)] = float('inf') 
        return torch.min(mat_masked, dim=1)
        
    def set_stats(self, hardest_positive_dist, hardest_negative_dist):
        pos_func = torch.min if self.use_similarity else torch.max
        neg_func = torch.max if self.use_similarity else torch.min
        self.hardest_triplet_dist = pos_func(hardest_positive_dist - hardest_negative_dist).item()
        self.hardest_pos_pair_dist = pos_func(hardest_positive_dist).item()
        self.hardest_neg_pair_dist = neg_func(hardest_negative_dist).item()