#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f

class BaseWeightRegularizer(torch.nn.Module):
    def __init__(self, normalize_weights=True, collect_stats=True):
        super().__init__()
        self.normalize_weights = normalize_weights
        self.collect_stats = collect_stats
        self.add_to_recordable_attributes(name="avg_weight_norm", is_stat=True, optional=True)

    def compute_loss(self, weights):
        raise NotImplementedError

    def forward(self, weights):
        """
        weights should have shape (num_classes, embedding_size)
        """
        c_f.reset_stats(self)
        if self.normalize_weights:
            weights = torch.nn.functional.normalize(weights, p=2, dim=1)
        self.weight_norms = torch.norm(weights, p=2, dim=1)
        self.avg_weight_norm = torch.mean(self.weight_norms)
        loss = self.compute_loss(weights)
        if loss == 0:
            loss = torch.sum(weights*0)
        return loss

    def add_to_recordable_attributes(self, name=None, list_of_names=None, is_stat=False, optional=False):
        if not optional or self.collect_stats: 
            c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names, is_stat=is_stat)