import torch
from torch import Tensor

from ..utils._common import accumulation_dtype
from .base_distance import BaseDistance


def _snr(query: Tensor, reference: Tensor) -> Tensor:
    dtype = torch.promote_types(query.dtype, reference.dtype)
    query = query.to(accumulation_dtype(query.dtype))
    reference = reference.to(accumulation_dtype(reference.dtype))
    ratio = torch.var(query - reference, dim=-1) / torch.var(query, dim=-1)
    return ratio.to(dtype)


class SNRDistance(BaseDistance):
    def compute_mat(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        return _snr(query_emb.unsqueeze(1), ref_emb)

    def pairwise_distance(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        return _snr(query_emb, ref_emb)
