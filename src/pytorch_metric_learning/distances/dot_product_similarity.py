import torch

from .base_distance import BaseDistance


class DotProductSimilarity(BaseDistance):
    is_inverted = True

    def compute_mat(self, query_emb, ref_emb):
        return torch.matmul(query_emb, ref_emb.transpose(-1, -2))

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.sum(query_emb * ref_emb, dim=1)
