from .base_weight_regularizer import BaseWeightRegularizer
import torch
from ..utils import common_functions as c_f

# modified from http://kaizhao.net/regularface
class RegularFaceRegularizer(BaseWeightRegularizer):
  
    def compute_loss(self, weights):
        num_classes = weights.size(0)
        cos = torch.mm(weights, weights.t())
        if not self.normalize_weights:
            norms = self.weight_norms.unsqueeze(1)
            cos = cos / (norms*norms.t())

        cos1 = cos.clone()
        with torch.no_grad():
            row_nums = torch.arange(num_classes).long().to(weights.device)
            cos1[row_nums, row_nums] = -float('inf')
            _, indices = torch.max(cos1, dim=1)
            mask = torch.zeros((num_classes, num_classes)).to(weights.device)
            mask[row_nums, indices] = 1
        
        return {"loss": {"losses": torch.sum(cos*mask, dim=1), "indices": c_f.torch_arange_from_size(weights), "reduction_type": "element"}}