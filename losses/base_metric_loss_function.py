#! /usr/bin/env python3

import torch


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
        learnable_param_names=None
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.num_class_per_param = num_class_per_param
        self.learnable_param_names = learnable_param_names
        self.initialize_learnable_parameters()

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
        labels = labels.to(embeddings.device)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        loss = self.compute_loss(embeddings, labels, indices_tuple)
        return loss

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
