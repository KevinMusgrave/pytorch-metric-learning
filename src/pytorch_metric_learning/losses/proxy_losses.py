#! /usr/bin/env python3

from .nca_loss import NCALoss
from .weight_regularizer_mixin import WeightRegularizerMixin
from ..utils import loss_and_miner_utils as lmu
import torch

class ProxyNCALoss(WeightRegularizerMixin, NCALoss):
    def __init__(self, num_classes, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, embedding_size))
        self.proxy_labels = torch.arange(num_classes)
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        if self.normalize_embeddings:
            prox = torch.nn.functional.normalize(self.proxies, p=2, dim=1)
        else:
            prox = self.proxies
        loss_dict = self.nca_computation(embeddings, prox, labels, self.proxy_labels.to(labels.device), indices_tuple)
        loss_dict["reg_loss"] = {"losses": self.regularization_loss(self.proxies), "indices": None, "reduction_type": "already_reduced"}
        return loss_dict
