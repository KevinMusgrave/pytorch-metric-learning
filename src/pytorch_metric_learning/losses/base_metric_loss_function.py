#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f
from ..reducers import MeanReducer

class BaseMetricLossFunction(torch.nn.Module):
    def __init__(
        self,
        normalize_embeddings=True,
        reducer=None
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.reducer = self.get_default_reducer() if reducer is None else reducer
        self.add_to_recordable_attributes(name="avg_embedding_norm")

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
        Returns: the loss (float)
        """
        c_f.assert_embeddings_and_labels_are_same_size(embeddings, labels)
        labels = labels.to(embeddings.device)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        self.embedding_norms = torch.norm(embeddings, p=2, dim=1)
        self.avg_embedding_norm = torch.mean(self.embedding_norms)

        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        return self.reducer(loss_dict, embeddings, labels)

    def add_to_recordable_attributes(self, name=None, list_of_names=None):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names)

    def zero_loss(self):
        return (0, None, None)

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def get_default_reducer(self):
        return MeanReducer()

    def sub_loss_names(self):
        return ["loss"]


class MultipleLosses(torch.nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = torch.nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1]*len(self.losses)

    def forward(self, embeddings, labels, indices_tuple=None):
        total_loss = 0
        for i, loss in enumerate(self.losses):
            total_loss += loss(embeddings, labels, indices_tuple)*self.weights[i]
        return total_loss