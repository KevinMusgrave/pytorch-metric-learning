from .base_weight_regularizer import BaseWeightRegularizer
import torch
from ..utils import common_functions as c_f

class CenterInvariantRegularizer(BaseWeightRegularizer):
    def compute_loss(self, weights):
        squared_weight_norms = self.distance.get_norm(weights)**2
        deviations_from_mean = squared_weight_norms - torch.mean(squared_weight_norms)
        return {"loss": {"losses": (deviations_from_mean**2) / 4, "indices": c_f.torch_arange_from_size(weights), "reduction_type": "element"}}