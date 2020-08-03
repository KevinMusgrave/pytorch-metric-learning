from .base_distance import BaseDistance
import torch
from ..utils import loss_and_miner_utils as lmu

class LpDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        if ref_emb is None:
            ref_emb = query_emb
        if torch.float16 in [query_emb.dtype, ref_emb.dtype]:
            return lmu.manual_dist_mat(query_emb, ref_emb) # half precision not supported by cdist
        else:
            return torch.cdist(query_emb, ref_emb)

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)