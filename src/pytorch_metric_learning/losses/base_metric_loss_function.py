#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f
from ..utils.module_with_records_and_reducer import ModuleWithRecordsAndReducer

class BaseMetricLossFunction(ModuleWithRecordsAndReducer):
    def __init__(self, normalize_embeddings=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.add_to_recordable_attributes(name="avg_embedding_norm", is_stat=True)

    def compute_loss(self, embeddings, labels, indices_tuple=None):
        """
        This has to be implemented and is what actually computes the loss.
        """
        raise NotImplementedError

    def forward(self, embeddings, labels, indices_tuple=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        c_f.assert_embeddings_and_labels_are_same_size(embeddings, labels)
        labels = labels.to(embeddings.device)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        self.embedding_norms = torch.norm(embeddings, p=2, dim=1)
        self.avg_embedding_norm = torch.mean(self.embedding_norms)

        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        return self.reducer(loss_dict, embeddings, labels)

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}


class MultipleLosses(torch.nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.is_dict = isinstance(losses, dict)
        self.losses = torch.nn.ModuleDict(losses) if self.is_dict else torch.nn.ModuleList(losses)
        if weights is not None:
            assert isinstance(weights, dict) if self.is_dict else c_f.is_list_or_tuple(weights)
            self.weights = weights
        else:
            self.weights = {k:1 for k in self.losses.keys()} if self.is_dict else [1]*len(losses)


    def forward(self, embeddings, labels, indices_tuple=None):
        total_loss = 0
        iterable = self.losses.items() if self.is_dict else enumerate(self.losses)
        if self.is_dict:
            for i, loss_func in iterable:
                total_loss += loss_func(embeddings, labels, indices_tuple)*self.weights[i]
        return total_loss