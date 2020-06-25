from .base_distance import BaseDistance
import torch

class LpDistance(BaseDistance):
    def __init__(self, p=2, power=1, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.power =  power

    def compute_mat(self, query_emb, ref_emb):
        mat = torch.cdist(query_emb, ref_emb, p=self.p)
        if self.power != 1:
            mat = mat**self.power
        return mat