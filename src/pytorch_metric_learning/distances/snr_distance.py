from .base_distance import BaseDistance
import torch

# Signal to Noise Ratio
class SNRDistance(BaseDistance):
    def compute_mat(self, query_emb, ref_emb):
        anchor_variances = torch.var(query_emb, dim=1)
        pairwise_diffs = query_emb.unsqueeze(1) - ref_emb
        pairwise_variances = torch.var(pairwise_diffs, dim=2)
        SNRs = pairwise_variances / (anchor_variances.unsqueeze(1))
        return SNRs