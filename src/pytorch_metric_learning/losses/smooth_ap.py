import torch
from torch import Tensor

from ..types import IndicesTuple, LossDict
import torch.nn.functional as F

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class SmoothAPLoss(BaseMetricLossFunction):
    def __init__(self, temperature=0.01, **kwargs):
        super().__init__(**kwargs)
        c_f.check_distance_type(self, CosineSimilarity)
        self.temperature = temperature

    def get_default_distance(self):
        return CosineSimilarity()

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        sims = self.distance(embeddings)
        others = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        same = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(sims.dtype)
        differences = sims.unsqueeze(1) - sims.unsqueeze(2)
        above = F.sigmoid(differences / self.temperature) * others
        all_ranks = 1 + above.sum(dim=-1)
        positive_ranks = 1 + (above * same.unsqueeze(1)).sum(dim=-1)
        ap = (positive_ranks / all_ranks * same).sum(dim=1) / same.sum(dim=1)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=ap.dtype)
        loss = (1 - ap) * miner_weights
        return {
            "ap_loss": {
                "losses": loss,
                "indices": c_f.torch_arange_from_size(loss),
                "reduction_type": "element",
            }
        }
