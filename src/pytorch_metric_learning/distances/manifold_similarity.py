import torch

from . import DotProductSimilarity


class ManifoldSimilarity(DotProductSimilarity):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        query_proxies = torch.stack([query_emb, ref_emb], dim=1)
        super().compute_mat(query_proxies, query_proxies)
