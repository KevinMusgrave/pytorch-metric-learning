#! /usr/bin/env python3


import torch
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class GenericPairLoss(BaseMetricLossFunction):
    """
    The function pair_based_loss has to be implemented by the child class.
    By default, this class extracts every positive and negative pair within a
    batch (based on labels) and passes the pairs to the loss function.
    The pairs can be passed to the loss function all at once (self.loss_once)
    or pairs can be passed iteratively (self.loss_loop) by going through each
    sample in a batch, and selecting just the positive and negative pairs
    containing that sample.
    Args:
        use_similarity: set to True if the loss function uses pairwise similarity
                        (dot product of each embedding pair). Otherwise,
                        euclidean distance will be used
        iterate_through_loss: set to True to use self.loss_loop and False otherwise
        squared_distances: if True, then the euclidean distance will be squared.
    """

    def __init__(
        self, use_similarity, mat_based_loss, squared_distances=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.loss_method = self.mat_based_loss if mat_based_loss else self.pair_based_loss
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        mat = lmu.get_pairwise_mat(embeddings, embeddings, self.use_similarity, self.squared_distances)
        if self.use_similarity and not self.normalize_embeddings:
            embedding_norms_mat = self.embedding_norms.unsqueeze(0)*self.embedding_norms.unsqueeze(1)
            mat = mat / (embedding_norms_mat)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels)
        return self.loss_method(mat, labels, indices_tuple)

    def _compute_loss(self):
        raise NotImplementedError

    def mat_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask)

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)
