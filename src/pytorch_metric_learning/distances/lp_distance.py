import torch
from torch import Tensor

from ..utils._common import LOW_PRECISION, accumulation_dtype
from .base_distance import BaseDistance


class LpDistance(BaseDistance):
    def compute_mat(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        if query_emb.dtype in LOW_PRECISION:
            mat = torch.cdist(
                query_emb.float(),
                ref_emb.float(),
                p=self.p,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            return mat.to(dtype=query_emb.dtype)
        return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        dtype = accumulation_dtype(query_emb.dtype)
        difference = query_emb.to(dtype) - ref_emb.to(dtype)
        distances = torch.linalg.vector_norm(difference, ord=self.p, dim=-1)
        return distances.to(query_emb.dtype)
