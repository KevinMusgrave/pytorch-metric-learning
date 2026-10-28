import torch
from torch import Tensor

from ..types import IndicesTuple, LossDict
from torch import nn

from ..utils import common_functions as c_f
from .base_metric_loss_function import BaseMetricLossFunction


class P2SGradLoss(BaseMetricLossFunction):
    def __init__(self, descriptors_dim, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.descriptors_dim = descriptors_dim
        self.num_classes = num_classes

        weight = torch.empty(descriptors_dim, num_classes).uniform_(-1, 1)
        self.weight = nn.Parameter(weight.renorm_(2, 1, 1e-5).mul_(1e5))

        self.m_loss = nn.MSELoss()

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.indices_tuple_not_supported(indices_tuple)

        c_f.check_parameter_placement(self, self.weight, embeddings)
        with torch.no_grad():
            self.weight.renorm_(2, 1, 1e-5).mul_(1e5)
        weight = self.weight.clone()
        dtype = embeddings.dtype

        rtol = 1e-2 if dtype == torch.float16 else 1e-5
        w_modulus = weight.pow(2).sum(0).pow(0.5)
        if not torch.all(torch.abs(w_modulus - 1.0) < rtol):
            raise RuntimeError(
                "P2SGrad weight columns must have unit norm after normalization"
            )

        x_modulus = embeddings.pow(2).sum(1).pow(0.5)
        cos_theta = embeddings.mm(weight)
        cos_theta = cos_theta / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        with torch.no_grad():
            index = torch.zeros_like(cos_theta)
            index.scatter_(1, labels.view(-1, 1), 1)

        loss = self.m_loss(cos_theta, index)

        return {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }
