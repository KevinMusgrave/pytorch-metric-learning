#! /usr/bin/env python3

from .base_miner import BasePostGradientMiner
import torch


class EmbeddingsAlreadyPackagedAsTriplets(BasePostGradientMiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # If the embeddings are grouped by triplet,
    # then use this miner to force the loss function to use the already-formed triplets
    def mine(self, embeddings, labels):
        batch_size = embeddings.size(0)
        a = torch.arange(0, batch_size, 3)
        p = torch.arange(1, batch_size, 3)
        n = torch.arange(2, batch_size, 3)
        return a, p, n
