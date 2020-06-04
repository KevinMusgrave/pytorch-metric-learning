#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f
from ..reducers import MeanReducer

class BaseMetricLossFunction(torch.nn.Module):
    """
    All loss functions extend this class
    Args:
        normalize_embeddings: type boolean. If True then normalize embeddins
                                to have norm = 1 before computing the loss
        num_class_per_param: type int. The number of classes for each parameter.
                            If your parameters don't have a separate value for each class,
                            then leave this at None
        learnable_param_names: type list of strings. Each element is the name of
                            attributes that should be converted to nn.Parameter 
    """
    def __init__(
        self,
        normalize_embeddings=True,
        num_class_per_param=None,
        learnable_param_names=None,
        reducer=None
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.num_class_per_param = num_class_per_param
        self.learnable_param_names = learnable_param_names
        self.reducer = MeanReducer() if reducer is None else reducer
        self.initialize_learnable_parameters()
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

        losses, loss_indices = self.compute_loss(embeddings, labels, indices_tuple)
        self.assert_losses_size(losses, loss_indices)
        return self.reducer(losses, loss_indices, labels)

    def initialize_learnable_parameters(self):
        """
        To learn hyperparams, create an attribute called learnable_param_names.
        This should be a list of strings which are the names of the
        hyperparameters to be learned
        """
        if self.learnable_param_names is not None:
            for k in self.learnable_param_names:
                v = getattr(self, k)
                setattr(self, k, self.create_learnable_parameter(v))

    def create_learnable_parameter(self, init_value, unsqueeze=False):
        """
        Returns nn.Parameter with an initial value of init_value
        and size of num_labels
        """
        vec_len = self.num_class_per_param if self.num_class_per_param else 1
        if unsqueeze:
            vec_len = (vec_len, 1)
        p = torch.nn.Parameter(torch.ones(vec_len) * init_value)
        return p

    def maybe_mask_param(self, param, labels):
        """
        This returns the hyperparameters corresponding to class labels (if applicable).
        If there is a hyperparameter for each class, then when computing the loss,
        the class hyperparameter has to be matched to the corresponding embedding.
        """
        if self.num_class_per_param:
            return param[labels]
        return param

    def add_to_recordable_attributes(self, name=None, list_of_names=None):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names)

    def element_indices(self, embeddings):
        return torch.arange(len(embeddings)).to(embeddings.device)

    def create_zero_loss(self, embeddings):
        losses = torch.sum(embeddings*0, dim=1)
        loss_indices = self.element_indices(embeddings)
        return losses, loss_indices



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