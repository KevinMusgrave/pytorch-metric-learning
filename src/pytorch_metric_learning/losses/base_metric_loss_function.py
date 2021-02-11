#! /usr/bin/env python3

import inspect

import torch

from ..utils import common_functions as c_f
from ..utils.module_with_records_and_reducer import ModuleWithRecordsReducerAndDistance
from .mixins import EmbeddingRegularizerMixin


class BaseMetricLossFunction(
    EmbeddingRegularizerMixin, ModuleWithRecordsReducerAndDistance
):
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
        labels = c_f.to_device(labels, embeddings)
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def _sub_loss_names(self):
        return ["loss"]

    def sub_loss_names(self):
        return self._sub_loss_names() + self.all_regularization_loss_names()

    def all_regularization_loss_names(self):
        reg_names = []
        for base_class in inspect.getmro(self.__class__):
            base_class_name = base_class.__name__
            mixin_keyword = "RegularizerMixin"
            if base_class_name.endswith(mixin_keyword):
                descriptor = base_class_name.replace(mixin_keyword, "").lower()
                if getattr(self, "{}_regularizer".format(descriptor)):
                    reg_names.extend(base_class.regularization_loss_names(self))
        return reg_names


class MultipleLosses(torch.nn.Module):
    def __init__(self, losses, miners=None, weights=None):
        super().__init__()
        self.is_dict = isinstance(losses, dict)
        self.losses = (
            torch.nn.ModuleDict(losses) if self.is_dict else torch.nn.ModuleList(losses)
        )

        if miners is not None:
            self.assertions_if_not_none(miners, match_all_keys=False)
            self.miners = (
                torch.nn.ModuleDict(miners)
                if self.is_dict
                else torch.nn.ModuleList(miners)
            )
        else:
            self.miners = None

        if weights is not None:
            self.assertions_if_not_none(weights, match_all_keys=True)
            self.weights = weights
        else:
            self.weights = (
                {k: 1 for k in self.losses.keys()}
                if self.is_dict
                else [1] * len(losses)
            )

    def forward(self, embeddings, labels, indices_tuple=None):
        if self.miners:
            assert indices_tuple is None
        total_loss = 0
        iterable = self.losses.items() if self.is_dict else enumerate(self.losses)
        for i, loss_func in iterable:
            curr_indices_tuple = self.get_indices_tuple(
                i, embeddings, labels, indices_tuple
            )
            total_loss += (
                loss_func(embeddings, labels, curr_indices_tuple) * self.weights[i]
            )
        return total_loss

    def get_indices_tuple(self, i, embeddings, labels, indices_tuple):
        if self.miners:
            if (self.is_dict and i in self.miners) or (
                not self.is_dict and self.miners[i] is not None
            ):
                indices_tuple = self.miners[i](embeddings, labels)
        return indices_tuple

    def assertions_if_not_none(self, x, match_all_keys):
        if x is not None:
            if self.is_dict:
                assert isinstance(x, dict)
                if match_all_keys:
                    assert sorted(list(x.keys())) == sorted(list(self.losses.keys()))
                else:
                    assert all(k in self.losses.keys() for k in x.keys())
            else:
                assert c_f.is_list_or_tuple(x)
                assert len(x) == len(self.losses)
