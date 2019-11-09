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
        self, use_similarity, iterate_through_loss, squared_distances=False, **kwargs
    ):
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.loss_method = self.loss_loop if iterate_through_loss else self.loss_once
        super().__init__(**kwargs)

    def compute_loss(self, embeddings, labels, indices_tuple):
        mat = (
            lmu.sim_mat(embeddings)
            if self.use_similarity
            else lmu.dist_mat(embeddings, squared=self.squared_distances)
        )
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels)
        return self.loss_method(mat, labels, indices_tuple)

    def pair_based_loss(
        self, pos_pairs, neg_pairs, pos_pair_anchor_labels, neg_pair_anchor_labels
    ):
        raise NotImplementedError

    def loss_loop(self, mat, labels, indices_tuple):
        loss = torch.tensor(0.0).to(mat.device)
        n = 0
        (a1_indices, p_indices, a2_indices, n_indices) = indices_tuple
        for i in range(mat.size(0)):
            pos_pair, neg_pair = [], []
            if len(a1_indices) > 0:
                p_idx = a1_indices == i
                pos_pair = mat[a1_indices[p_idx], p_indices[p_idx]]
            if len(a2_indices) > 0:
                n_idx = a2_indices == i
                neg_pair = mat[a2_indices[n_idx], n_indices[n_idx]]
            loss += self.pair_based_loss(
                pos_pair, neg_pair, labels[a1_indices[p_idx]], labels[a2_indices[n_idx]]
            )
            n += 1
        return loss / (n if n > 0 else 1)

    def loss_once(self, mat, labels, indices_tuple):
        (a1_indices, p_indices, a2_indices, n_indices) = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1_indices) > 0:
            pos_pair = mat[a1_indices, p_indices]
        if len(a2_indices) > 0:
            neg_pair = mat[a2_indices, n_indices]
        return self.pair_based_loss(
            pos_pair, neg_pair, labels[a1_indices], labels[a2_indices]
        )
