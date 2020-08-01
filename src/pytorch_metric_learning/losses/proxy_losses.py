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
        
    def cast_types(self, dtype, device):
        self.proxies.data = self.proxies.data.to(device).type(dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        loss_dict = self.nca_computation(embeddings, self.proxies, labels, self.proxy_labels.to(labels.device), indices_tuple)
        loss_dict["reg_loss"] = self.regularization_loss(self.proxies)
        return loss_dict

    def sub_loss_names(self):
        return ["loss", "reg_loss"]