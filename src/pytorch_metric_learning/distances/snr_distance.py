import torch
from torch import Tensor

from .base_distance import BaseDistance


class SNRDistance(BaseDistance):
    def compute_mat(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        anchor_variances = torch.var(query_emb, dim=1)
        pairwise_diffs = query_emb.unsqueeze(1) - ref_emb
        pairwise_variances = torch.var(pairwise_diffs, dim=2)
        return pairwise_variances / (anchor_variances.unsqueeze(1))

    def pairwise_distance(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        query_var = torch.var(query_emb, dim=1)
        query_ref_var = torch.var(query_emb - ref_emb, dim=1)
        return query_ref_var / query_var
