#! /usr/bin/env python3


from .base_miner import BaseSubsetBatchMiner
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
import numpy as np
import torch

class MaximumLossMiner(BaseSubsetBatchMiner):
    def __init__(self, loss, miner=None, num_trials=5, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
        self.miner = miner
        self.num_trials = num_trials
        self.add_to_recordable_attributes(list_of_names=["avg_loss", "max_loss"], is_stat=True)

    def mine(self, embeddings, labels, *_):
        losses = []
        all_subset_idx = []
        for i in range(self.num_trials):
            rand_subset_idx = c_f.NUMPY_RANDOM.choice(len(embeddings), size=self.output_batch_size, replace=False)
            rand_subset_idx = torch.from_numpy(rand_subset_idx)
            all_subset_idx.append(rand_subset_idx)
            curr_embeddings, curr_labels = embeddings[rand_subset_idx], labels[rand_subset_idx]
            indices_tuple = self.inner_miner(curr_embeddings, curr_labels)
            losses.append(self.loss(curr_embeddings, curr_labels, indices_tuple).item())
        max_idx = np.argmax(losses)
        self.avg_loss = np.mean(losses)
        self.max_loss = losses[max_idx]
        return all_subset_idx[max_idx]

    def inner_miner(self, embeddings, labels):
        if self.miner:
            return self.miner(embeddings, labels)
        return None