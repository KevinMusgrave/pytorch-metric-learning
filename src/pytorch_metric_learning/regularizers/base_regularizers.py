#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f
from ..utils.module_with_records_and_reducer import ModuleWithRecordsReducerAndDistance

class BaseWeightRegularizer(ModuleWithRecordsReducerAndDistance):
    def compute_loss(self, weights):
        raise NotImplementedError

    def forward(self, weights):
        """
        weights should have shape (num_classes, embedding_size)
        """
        self.reset_stats()
        loss_dict = self.compute_loss(weights)
        return self.reducer(loss_dict, weights, c_f.torch_arange_from_size(weights))


class BaseEmbeddingRegularizer(ModuleWithRecordsReducerAndDistance):
    def compute_loss(self, embeddings):
        raise NotImplementedError

    def forward(self, embeddings, labels, indices_tuple):
        self.reset_stats()
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        return self.reducer(loss_dict, embeddings, labels)