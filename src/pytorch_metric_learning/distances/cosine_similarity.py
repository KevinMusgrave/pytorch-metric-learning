from .base_distance import BaseDistance
import torch

class CosineSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        ref_emb = torch.nn.functional.normalize(ref_emb, p=2, dim=1)
        return torch.matmul(query_emb, ref_emb.t())