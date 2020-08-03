from .base_regularizer import BaseRegularizer
import torch
from ..utils import common_functions as c_f

class LpRegularizer(BaseRegularizer):
    def __init__(self, p=2, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def compute_loss(self, embeddings):
        l2_reg = torch.norm(embeddings, p=self.p, dim=1)
        return {"loss": {"losses": l2_reg, "indices": c_f.torch_arange_from_size(embeddings), "reduction_type": "element"}}