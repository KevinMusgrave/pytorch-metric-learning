from .base_miner import BaseTupleMiner
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f

class BatchHardMiner(BaseTupleMiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(list_of_names=["hardest_triplet_dist", "hardest_pos_pair_dist", "hardest_neg_pair_dist"],
                                            is_stat=True)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, ref_labels)

        pos_func = self.get_min_per_row if self.distance.is_inverted else self.get_max_per_row
        neg_func = self.get_max_per_row if self.distance.is_inverted else self.get_min_per_row

        (hardest_positive_dist, hardest_positive_indices), a1p_keep = pos_func(mat, a1_idx, p_idx)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = neg_func(mat, a2_idx, n_idx)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        self.set_stats(hardest_positive_dist[a_keep_idx], hardest_negative_dist[a_keep_idx])
        a = torch.arange(mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]        
        return a, p, n 

    def get_max_per_row(self, mat, anchor_idx, other_idx):
        mask = torch.zeros_like(mat)
        mask[anchor_idx, other_idx] = 1
        non_zero_rows = torch.any(mask!=0, dim=1)
        mat_masked = mat * mask 
        return torch.max(mat_masked, dim=1), non_zero_rows

    def get_min_per_row(self, mat, anchor_idx, other_idx):
        pos_inf = c_f.pos_inf(mat.dtype)
        mask = torch.ones_like(mat) * pos_inf
        mask[anchor_idx, other_idx] = 1
        non_inf_rows = torch.any(mask!=pos_inf, dim=1)
        mat = mat.clone()
        mat[mask==pos_inf] = pos_inf
        return torch.min(mat, dim=1), non_inf_rows
        
    def set_stats(self, hardest_positive_dist, hardest_negative_dist):
        if self.collect_stats:
            with torch.no_grad():
                pos_func = torch.min if self.distance.is_inverted else torch.max
                neg_func = torch.max if self.distance.is_inverted else torch.min
                try:
                    self.hardest_triplet_dist = pos_func(hardest_positive_dist - hardest_negative_dist).item()
                    self.hardest_pos_pair_dist = pos_func(hardest_positive_dist).item()
                    self.hardest_neg_pair_dist = neg_func(hardest_negative_dist).item()
                except RuntimeError:
                    self.hardest_triplet_dist = 0
                    self.hardest_pos_pair_dist = 0
                    self.hardest_neg_pair_dist = 0