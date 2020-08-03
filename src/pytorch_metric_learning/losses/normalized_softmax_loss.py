from .regularizer_mixins import WeightRegularizerMixin
from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from ..distances import DotProductSimilarity

class NormalizedSoftmaxLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    def __init__(self, temperature, embedding_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.W = torch.nn.Parameter(torch.randn(embedding_size, num_classes))
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        
    def cast_types(self, dtype, device):
        self.W.data = self.W.data.to(device).type(dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        normalized_W = torch.nn.functional.normalize(self.W, p=2, dim=0)
        exponent = self.distance(embeddings, normalized_W.t()) / self.temperature
        if not self.distance.is_inverted:
            exponent = -exponent
        unweighted_loss = self.cross_entropy(exponent, labels)
        miner_weighted_loss = unweighted_loss*miner_weights
        loss_dict = {"loss": {"losses": miner_weighted_loss, "indices": c_f.torch_arange_from_size(embeddings), "reduction_type": "element"}}
        self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
        return loss_dict

    def get_default_distance(self):
        return DotProductSimilarity()