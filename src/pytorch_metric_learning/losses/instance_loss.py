import torch
from torch import Tensor

from ..types import IndicesTuple, LossDict
import torch.nn.functional as F

from ..distances import CosineSimilarity
from ..reducers import MeanReducer
from ..utils import common_functions as c_f
from .base_metric_loss_function import BaseMetricLossFunction


class InstanceLoss(BaseMetricLossFunction):
    def __init__(self, gamma=64, **kwargs):
        super().__init__(**kwargs)
        c_f.check_distance_type(self, CosineSimilarity)
        self.gamma = gamma

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        c_f.labels_required(labels)
        c_f.indices_tuple_not_supported(indices_tuple)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        sim1 = self.distance(embeddings) * self.gamma
        _, sim_label = torch.unique(labels, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label, reduction="none")
        return {
            "loss": {
                "losses": loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }

    def get_default_reducer(self):
        return MeanReducer()

    def get_default_distance(self):
        return CosineSimilarity()
