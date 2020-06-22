import torch
from ..utils import common_functions as c_f
from ..utils.module_with_records import ModuleWithRecords


class BaseDistance(ModuleWithRecords):
    def __init__(self, small_values_for_large_similarity=True, **kwargs):
        super().__init__(**kwargs)
        self.small_values_for_large_similarity = small_values_for_large_similarity

    # def forward(self, query_emb, ref_emb, indices_tuple=None):
    #     if self.should_compute_dist_mat(query_emb, ref_emb, indices_tuple):
    #         distances = self.get_dist_mat(query_emb, ref_emb)
    #     else:
    #         distances = self.get_dist_for_indices(query_emb, ref_emb, indices_tuple)
    #     assert distances.size() == torch.Size([query_emb.size(0), ref_emb.size(1)])
    #     return distances

    # def should_compute_dist_mat(self, query_emb, ref_emb, indices_tuple):
    #     if indices_tuple is None:
    #         return True
    #     dist_mat_comp = len(query_emb) * len(ref_emb)
    #     if len(indices_tuple) == 3:
    #         a, _, _ = indices_tuple
    #         if len(a) < dist_mat_comp
    #             return False
    #     if len(indices_tuple) == 4:
    #         a1, _, a2, _ = indices_tuple
    #         if (len(a1) + len(a2)) < dist_mat_comp:
    #             return False
    #     return True

    # def get_dist_mat(self, query_emb, ref_emb):
    #     raise NotImplementedError

    # def get_dist_for_indices(self, query_emb, ref_emb, indices_tuple):
    #     raise NotImplementedError