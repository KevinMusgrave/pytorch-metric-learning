#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f
from ..reducers import MeanReducer

class BaseWeightRegularizer(torch.nn.Module):
    def __init__(self, normalize_weights=True, reducer=None, collect_stats=True):
        super().__init__()
        self.normalize_weights = normalize_weights
        self.reducer = self.get_default_reducer() if reducer is None else reducer
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
        loss_dict = self.compute_loss(weights)
        return self.reducer(loss_dict, weights, c_f.torch_arange_from_size(weights))

    def add_to_recordable_attributes(self, name=None, list_of_names=None, is_stat=False, optional=False):
        if not optional or self.collect_stats: 
            c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names, is_stat=is_stat)

    def get_default_reducer(self):
        return MeanReducer()

    def get_reducer(self):
        reducer = self.get_default_reducer()
        if isinstance(reducer, MultipleReducers) or len(self.sub_loss_names()) == 1:
            return reducer
        return MultipleReducers({k:self.get_default_reducer() for k in self.sub_loss_names()})

    def sub_loss_names(self):
        return ["loss"]