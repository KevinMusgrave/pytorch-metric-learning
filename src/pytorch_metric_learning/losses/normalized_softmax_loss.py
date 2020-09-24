from .mixins import WeightRegularizerMixin
from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from ..distances import DotProductSimilarity


class NormalizedSoftmaxLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    def __init__(self, num_classes, embedding_size, temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.W = torch.nn.Parameter(torch.Tensor(embedding_size, num_classes))
        self.weight_init_func(self.W)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "num_classes", "temperature"],
            is_stat=False,
        )

    def cast_types(self, dtype, device):
        self.W.data = self.W.data.to(device).type(dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        normalized_W = self.distance.normalize(self.W, dim=0)
        exponent = self.distance(embeddings, normalized_W.t()) / self.temperature
        if not self.distance.is_inverted:
            exponent = -exponent
        unweighted_loss = self.cross_entropy(exponent, labels)
        miner_weighted_loss = unweighted_loss * miner_weights
        loss_dict = {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
        return loss_dict

    def get_default_distance(self):
        return DotProductSimilarity()
