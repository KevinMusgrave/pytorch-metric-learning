from .base_regularizer import BaseRegularizer
import torch
from ..utils import common_functions as c_f


class LpRegularizer(BaseRegularizer):
    def __init__(self, p=2, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.add_to_recordable_attributes(list_of_names=["p"], is_stat=False)

    def compute_loss(self, embeddings):
        reg = torch.norm(embeddings, p=self.p, dim=1)
        return {
            "loss": {
                "losses": reg,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
