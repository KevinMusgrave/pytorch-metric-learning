import torch
from ..utils import common_functions as c_f
from ..utils.module_with_records import ModuleWithRecords


class BaseDistance(ModuleWithRecords):
    def __init__(self, normalize_embeddings=True, p=2, is_inverted=False, **kwargs):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.is_inverted = is_inverted
        self.add_to_recordable_attributes(name="avg_embedding_norm", is_stat=True)

    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        if ref_emb is None:
            ref_emb = query_emb
        if self.normalize_embeddings:
            query_emb = torch.nn.functional.normalize(query_emb, p=self.p, dim=1)
            ref_emb = torch.nn.functional.normalize(ref_emb, p=self.p, dim=1)
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

    def x_less_than_y(self, x, y, or_equal=False):
        condition = (x > y) if self.is_inverted else (x < y)
        if or_equal:
            condition |= x == y
        return condition

    def x_greater_than_y(self, x, y, or_equal=False):
        return ~self.x_less_than_y(x, y, not or_equal)

    # This measures how much bigger the neg distance is compared to the positive distance.
    def pos_neg_margin(self, pos, neg):
        if self.is_inverted:
            return pos - neg
        return neg - pos