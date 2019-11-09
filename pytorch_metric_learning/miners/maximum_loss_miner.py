#! /usr/bin/env python3


from .base_miner import BasePreGradientMiner
from ..utils import loss_and_miner_utils as lmu
import numpy as np
import torch

class MaximumLossMiner(BasePreGradientMiner):
    def __init__(self, loss_function, mining_function=None, num_trials=5, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = loss_function
        self.mining_function = mining_function
        self.num_trials = num_trials

    def mine(self, embeddings, labels):
        losses = []
        rand_subset_idx = torch.randint(0, len(embeddings), size=(self.num_trials, self.output_batch_size))
        for i in range(self.num_trials):
            curr_embeddings, curr_labels = embeddings[rand_subset_idx[i]], labels[rand_subset_idx[i]]
            indices_tuple = self.inner_miner(curr_embeddings, curr_labels)
            losses.append(self.loss_function(curr_embeddings, curr_labels, indices_tuple))
        max_loss_idx = np.argmax(losses)
        return rand_subset_idx[max_loss_idx]

    def inner_miner(self, embeddings, labels):
        if self.mining_function:
            return self.mining_function(embeddings, labels)
        return None