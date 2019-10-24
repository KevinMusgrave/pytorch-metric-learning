#! /usr/bin/env python3

from . import nca_loss as nca
import torch
from ..utils import loss_and_miner_utils as lmu

class ProxyNCALoss(nca.NCALoss):
    def __init__(self, num_labels, embedding_size, **kwargs):
        self.proxies = torch.nn.Parameter(torch.randn(num_labels, embedding_size))
        self.proxy_labels = torch.arange(num_labels)
        super().__init__(**kwargs)

    def compute_loss(self, embeddings, labels, *_):
        if self.normalize_embeddings:
            prox = torch.nn.functional.normalize(self.proxies, p=2, dim=1)
        else:
            prox = self.proxies
        return self.nca_computation(embeddings, prox, labels, self.proxy_labels.to(labels.device))