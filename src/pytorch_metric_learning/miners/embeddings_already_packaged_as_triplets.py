import torch

from .base_miner import BaseMiner


class EmbeddingsAlreadyPackagedAsTriplets(BaseMiner):
    def __init__(self):
        super().__init__()

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        batch_size = embeddings.size(0)
        a = torch.arange(0, batch_size, 3)
        p = torch.arange(1, batch_size, 3)
        n = torch.arange(2, batch_size, 3)
        return a, p, n
