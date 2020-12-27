import torch

from .base_distance import BaseDistance


class DotProductSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        return torch.matmul(query_emb, ref_emb.t())

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.sum(query_emb * ref_emb, dim=1)
