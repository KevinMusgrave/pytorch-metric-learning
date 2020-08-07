from .base_regularizer import BaseRegularizer
import torch
from ..utils import common_functions as c_f

class ZeroMeanRegularizer(BaseRegularizer):
    def compute_loss(self, embeddings):
        return {"loss": {"losses": torch.abs(torch.sum(embeddings, dim=1)), "indices": c_f.torch_arange_from_size(embeddings), "reduction_type": "element"}}