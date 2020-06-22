from .base_distance import BaseDistance
import torch
from ..utils import loss_and_miner_utils as lmu


class LpDistance(BaseDistance):
    def __init__(self, p, power=None, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.power = 1 if power is None else power

    def get_dist_mat(self, query_emb, ref_emb):
        if self.p == 2:
            if self.power == 2:
                return lmu.dist_mat(query_emb, ref_emb, squared=True)
            else:
                return lmu.dist_mat(query_emb, ref_emb, squared=False) ** self.power
            
    


    