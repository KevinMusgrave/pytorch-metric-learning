import torch
from ..utils import common_functions as c_f
from ..utils.module_with_records import ModuleWithRecords


class BaseDistance(ModuleWithRecords):
    def __init__(self, is_inverted=False, **kwargs):
        super().__init__(**kwargs)
        self.is_inverted = is_inverted

    def forward(self, query_emb, ref_emb=None):
        if ref_emb is None:
            ref_emb = query_emb
        mat = self.compute_mat(query_emb, ref_emb)
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        raise NotImplementedError

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.max(*args, **kwargs)
        return torch.min(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.min(*args, **kwargs)
        return torch.max(*args, **kwargs)    

    def x_more_similar_than_y(self, x, y, or_equal=False):
        condition = (x > y) if self.is_inverted else (x < y)
        if or_equal:
            condition &= x == y
        return condition

    def x_less_similar_than_y(self, x, y, or_equal=False):
        condition = (x < y) if self.is_inverted else (x > y)
        if or_equal:
            condition &= x == y
        return condition

    # This measures how much bigger the neg distance is compared to the positive distance.
    def pos_neg_margin(self, pos, neg):
        if self.is_inverted:
            return pos - neg
        return neg - pos