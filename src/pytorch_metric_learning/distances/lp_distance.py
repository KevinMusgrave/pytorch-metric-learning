from .base_distance import BaseDistance
import torch
from ..utils import loss_and_miner_utils as lmu

class LpDistance(BaseDistance):
    def __init__(self, power=1, **kwargs):
        super().__init__(**kwargs)
        self.power =  power
        self.add_to_recordable_attributes(list_of_names=["power"], is_stat=False)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        mat = lmu.dist_mat(query_emb, ref_emb)
        if self.power != 1:
            mat = mat**self.power
        return mat