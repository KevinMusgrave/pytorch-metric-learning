from .base_weight_regularizer import BaseWeightRegularizer
import torch

class CenterInvariantRegularizer(BaseWeightRegularizer):
    def __init__(self, normalize_weights=False):
        super().__init__(normalize_weights)
        assert not self.normalize_weights, "normalize_weights must be False for CenterInvariantRegularizer"
  
    def compute_loss(self, weights):
        squared_weight_norms = self.weight_norms**2
        deviations_from_mean = squared_weight_norms - torch.mean(squared_weight_norms)
        return torch.mean((deviations_from_mean**2) / 4)